#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <STEPControl_Reader.hxx>
#include <TopoDS_Shape.hxx>
#include <Bnd_Box.hxx>
#include <BRepBndLib.hxx>
#include <gp_Trsf.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <Aspect_DisplayConnection.hxx>
#include <OpenGl_GraphicDriver.hxx>
#include <V3d_Viewer.hxx>
#include <V3d_View.hxx>
#include <V3d_DirectionalLight.hxx>
#include <AIS_Shape.hxx>
#include <AIS_InteractiveContext.hxx>
#include <Prs3d_Drawer.hxx>
#include <Prs3d_LineAspect.hxx>
#include <Prs3d_IsoAspect.hxx>
#include <Image_AlienPixMap.hxx>
#include <Quantity_Color.hxx>
#include <Quantity_ColorRGBA.hxx>
#include <Xw_Window.hxx>

struct View {
    double yaw, pitch, radius, fov;
    std::string output_path;
};

struct Args {
    std::string input_path;
    std::string views_file;
    double yaw = 0, pitch = 0, radius = 2, fov = 0.698;
    std::string output_path;
};

std::vector<View> ParseViews(const std::string& path) {
    std::vector<View> views;
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open views file: " << path << std::endl;
        return views;
    }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    size_t pos = 0;
    auto skip = [&]() { while (pos < json.size() && isspace(json[pos])) pos++; };
    auto expect = [&](char c) { skip(); if (pos < json.size() && json[pos] == c) { pos++; return true; } return false; };
    auto parseString = [&]() -> std::string {
        skip();
        if (pos >= json.size() || json[pos] != '"') return "";
        pos++;
        std::string s;
        while (pos < json.size() && json[pos] != '"') { s += json[pos++]; }
        pos++;
        return s;
    };
    auto parseNumber = [&]() -> double {
        skip();
        size_t start = pos;
        if (pos < json.size() && (json[pos] == '-' || json[pos] == '+')) pos++;
        while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '.' || json[pos] == 'e' || json[pos] == 'E' || json[pos] == '-' || json[pos] == '+')) pos++;
        return std::stod(json.substr(start, pos - start));
    };

    expect('[');
    while (true) {
        skip();
        if (pos >= json.size() || json[pos] == ']') break;
        expect('{');
        View v{};
        while (true) {
            skip();
            if (pos >= json.size() || json[pos] == '}') break;
            std::string key = parseString();
            expect(':');
            if (key == "output") v.output_path = parseString();
            else if (key == "yaw") v.yaw = parseNumber();
            else if (key == "pitch") v.pitch = parseNumber();
            else if (key == "radius") v.radius = parseNumber();
            else if (key == "fov") v.fov = parseNumber();
            skip();
            if (pos < json.size() && json[pos] == ',') pos++;
        }
        expect('}');
        views.push_back(v);
        skip();
        if (pos < json.size() && json[pos] == ',') pos++;
    }
    return views;
}

TopoDS_Shape NormalizeShape(const TopoDS_Shape& shape) {
    Bnd_Box bbox;
    BRepBndLib::Add(shape, bbox);
    double xmin, ymin, zmin, xmax, ymax, zmax;
    bbox.Get(xmin, ymin, zmin, xmax, ymax, zmax);

    double max_dim = std::max({xmax - xmin, ymax - ymin, zmax - zmin});
    double scale = (max_dim > 1e-6) ? (1.0 / max_dim) : 1.0;

    gp_Pnt center((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0);
    gp_Trsf transform;
    transform.SetScale(center, scale);

    gp_Trsf translation;
    translation.SetTranslation(center, gp_Pnt(0, 0, 0));
    transform.Multiply(translation);

    return BRepBuilderAPI_Transform(shape, transform).Shape();
}

void RenderView(Handle(V3d_View)& view, const View& v,
                const Quantity_Color& bgColor, const std::string& outPath) {
    view->Camera()->SetProjectionType(Graphic3d_Camera::Projection_Perspective);
    view->Camera()->SetFOVy(v.fov * 180.0 / M_PI);
    view->SetEye(v.radius, v.radius, v.radius);
    view->SetAt(0, 0, 0);
    view->SetUp(0, 0, 1);
    view->Rotate(v.yaw, v.pitch, 0, 0, 0, Standard_True);

    view->FitAll();
    view->MustBeResized();
    view->Redraw();
    view->Update();

    Image_AlienPixMap img;
    if (!view->ToPixMap(img, 1024, 1024, Graphic3d_BT_RGBA)) {
        std::cerr << "Failed to render: " << outPath << std::endl;
        return;
    }

    Image_AlienPixMap imgRGBA;
    imgRGBA.InitZero(Image_Format_RGBA, img.Width(), img.Height());

    for (Standard_Size y = 0; y < img.Height(); ++y)
        for (Standard_Size x = 0; x < img.Width(); ++x)
            imgRGBA.SetPixelColor(x, y, img.PixelColor(x, y));

    for (Standard_Size y = 0; y < imgRGBA.Height(); ++y) {
        for (Standard_Size x = 0; x < imgRGBA.Width(); ++x) {
            Quantity_ColorRGBA color = imgRGBA.PixelColor(x, y);
            if (color.GetRGB().SquareDistance(bgColor) < 0.01)
                color.SetAlpha(0.0);
            else
                color.SetAlpha(1.0);
            imgRGBA.SetPixelColor(x, y, color);
        }
    }

    int stride = (int)imgRGBA.SizeRowBytes();
    if (stbi_write_png(outPath.c_str(), (int)imgRGBA.Width(), (int)imgRGBA.Height(), 4, imgRGBA.Data(), stride))
        std::cout << "Rendered: " << outPath << std::endl;
    else
        std::cerr << "Failed to save: " << outPath << std::endl;
}

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--object") args.input_path = argv[++i];
        else if (arg == "--views") args.views_file = argv[++i];
        else if (arg == "--output") args.output_path = argv[++i];
        else if (arg == "--yaw") args.yaw = std::stod(argv[++i]);
        else if (arg == "--pitch") args.pitch = std::stod(argv[++i]);
        else if (arg == "--radius") args.radius = std::stod(argv[++i]);
        else if (arg == "--fov") args.fov = std::stod(argv[++i]);
    }

    std::vector<View> views;
    if (!args.views_file.empty()) {
        views = ParseViews(args.views_file);
        if (views.empty()) { std::cerr << "No views loaded." << std::endl; return 1; }
    } else {
        views.push_back({args.yaw, args.pitch, args.radius, args.fov, args.output_path});
    }

    STEPControl_Reader reader;
    if (reader.ReadFile(args.input_path.c_str()) != IFSelect_RetDone) {
        std::cerr << "Error: Could not read file " << args.input_path << std::endl;
        return 1;
    }
    reader.TransferRoots();
    TopoDS_Shape shape = NormalizeShape(reader.OneShape());
    if (shape.IsNull()) {
        std::cerr << "Error: Shape is null after normalization." << std::endl;
        return 1;
    }

    Handle(Aspect_DisplayConnection) disp = new Aspect_DisplayConnection();
    Handle(OpenGl_GraphicDriver) driver = new OpenGl_GraphicDriver(disp);
    Handle(V3d_Viewer) viewer = new V3d_Viewer(driver);
    Handle(V3d_View) view = viewer->CreateView();
    Handle(AIS_InteractiveContext) context = new AIS_InteractiveContext(viewer);

    Handle(V3d_DirectionalLight) light = new V3d_DirectionalLight(V3d_Zneg, Quantity_NOC_WHITE, Standard_True);
    viewer->AddLight(light);
    viewer->SetLightOn();

    Handle(Xw_Window) win = new Xw_Window(disp, "Render", 0, 0, 1024, 1024);
    win->SetVirtual(Standard_True);
    win->Map();
    view->SetWindow(win);

    Quantity_Color bgColor(1.0, 0.0, 1.0, Quantity_TOC_RGB);
    view->SetBackgroundColor(bgColor);

    Handle(AIS_Shape) aisShape = new AIS_Shape(shape);
    aisShape->SetDisplayMode(AIS_Shaded);

    // Disable iso-lines
    aisShape->Attributes()->SetIsoOnTriangulation(Standard_False);
    aisShape->Attributes()->SetIsoOnPlane(Standard_False);
    aisShape->Attributes()->UIsoAspect()->SetNumber(0);
    aisShape->Attributes()->VIsoAspect()->SetNumber(0);

    // Enable face boundary edges in black
    aisShape->Attributes()->SetFaceBoundaryDraw(Standard_True);
    aisShape->Attributes()->SetFaceBoundaryAspect(new Prs3d_LineAspect(Quantity_NOC_BLACK, Aspect_TOL_SOLID, 1.0));

    context->Display(aisShape, Standard_False);

    for (size_t i = 0; i < views.size(); ++i) {
        std::cout << "[" << (i + 1) << "/" << views.size() << "] ";
        RenderView(view, views[i], bgColor, views[i].output_path);
    }

    return 0;
}