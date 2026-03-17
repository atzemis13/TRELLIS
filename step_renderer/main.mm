#include <iostream>
#include <string>
#include <vector>
#import <Cocoa/Cocoa.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// OCCT Includes
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
#include <Image_AlienPixMap.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_RectangularTrimmedSurface.hxx>
#include <TopoDS.hxx>
#include <TopExp_Explorer.hxx>
#include <BRep_Tool.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <Quantity_Color.hxx>
#include <Quantity_ColorRGBA.hxx>
#include <Cocoa_Window.hxx> // Required for macOS

struct CameraArgs {
    double yaw = 0.0, pitch = 0.0, radius = 5.0, fov = 45.0;
    std::string input_path, output_path;
};

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

Handle(Geom_Surface) GetBasisSurface(const Handle(Geom_Surface)& surf) {
    if (surf->IsKind(STANDARD_TYPE(Geom_RectangularTrimmedSurface))) {
        return GetBasisSurface(Handle(Geom_RectangularTrimmedSurface)::DownCast(surf)->BasisSurface());
    }
    return surf;
}

TopoDS_Shape ExtractKnots(const TopoDS_Shape& shape) {
    TopoDS_Compound compound;
    BRep_Builder builder;
    builder.MakeCompound(compound);

    TopExp_Explorer expl(shape, TopAbs_FACE);
    for (; expl.More(); expl.Next()) {
        const TopoDS_Face& face = TopoDS::Face(expl.Current());
        TopLoc_Location loc;
        Handle(Geom_Surface) surf = BRep_Tool::Surface(face, loc);
        
        if (surf.IsNull()) continue;
        surf = GetBasisSurface(surf);

        if (surf->IsKind(STANDARD_TYPE(Geom_BSplineSurface))) {
            Handle(Geom_BSplineSurface) bspline = Handle(Geom_BSplineSurface)::DownCast(surf);
            
            const TColStd_Array1OfReal& uKnots = bspline->UKnots();
            for (int i = uKnots.Lower(); i <= uKnots.Upper(); ++i) {
                Handle(Geom_Curve) curve = bspline->UIso(uKnots.Value(i));
                TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(curve);
                edge.Move(loc);
                builder.Add(compound, edge);
            }

            const TColStd_Array1OfReal& vKnots = bspline->VKnots();
            for (int i = vKnots.Lower(); i <= vKnots.Upper(); ++i) {
                Handle(Geom_Curve) curve = bspline->VIso(vKnots.Value(i));
                TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(curve);
                edge.Move(loc);
                builder.Add(compound, edge);
            }
        }
    }
    return compound;
}

int main(int argc, char** argv) {
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    [NSApplication sharedApplication];

    CameraArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--yaw") args.yaw = std::stod(argv[++i]);
        else if (arg == "--pitch") args.pitch = std::stod(argv[++i]);
        else if (arg == "--radius") args.radius = std::stod(argv[++i]);
        else if (arg == "--fov") args.fov = std::stod(argv[++i]);
        else if (arg == "--object") args.input_path = argv[++i];
        else if (arg == "--output") args.output_path = argv[++i];
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

    // macOS Virtual Window Setup
    Handle(Cocoa_Window) win = new Cocoa_Window("Render", 0, 0, 1024, 1024);
    win->SetVirtual(Standard_True);
    win->Map(); // Ensure window is mapped
    view->SetWindow(win);
    // Set background to Magenta for chroma keying
    Quantity_Color bgColor(1.0, 0.0, 1.0, Quantity_TOC_RGB);
    view->SetBackgroundColor(bgColor);

    Handle(AIS_Shape) aisShape = new AIS_Shape(shape);
    aisShape->SetDisplayMode(AIS_Shaded);

    // Enable edges
    aisShape->Attributes()->SetFaceBoundaryDraw(Standard_True);
    aisShape->Attributes()->SetFaceBoundaryAspect(new Prs3d_LineAspect(Quantity_NOC_BLACK, Aspect_TOL_SOLID, 1.0));

    context->Display(aisShape, Standard_False);

    // Extract and display knots
    TopoDS_Shape knots = ExtractKnots(shape);
    if (!knots.IsNull()) {
        Handle(AIS_Shape) aisKnots = new AIS_Shape(knots);
        aisKnots->SetColor(Quantity_NOC_BLUE);
        context->Display(aisKnots, Standard_False);
    }

    // Camera Logic
    view->Camera()->SetProjectionType(Graphic3d_Camera::Projection_Perspective);
    view->Camera()->SetFOVy(args.fov * 180.0 / M_PI);
    view->SetEye(args.radius, args.radius, args.radius);
    view->SetAt(0, 0, 0);
    view->Rotate(args.yaw, args.pitch, 0, 0, 0, Standard_True);
    
    view->FitAll(); // Force fit to view to ensure object is visible
    view->MustBeResized();
    view->Redraw(); // Force redraw
    view->Update();

    Image_AlienPixMap img;
    if (view->ToPixMap(img, 1024, 1024, Graphic3d_BT_RGBA)) {
        // Always convert to a new RGBA image to ensure we have full control over the buffer and format
        Image_AlienPixMap imgRGBA;
        imgRGBA.InitZero(Image_Format_RGBA, img.Width(), img.Height());
        
        for (Standard_Size y = 0; y < img.Height(); ++y) {
            for (Standard_Size x = 0; x < img.Width(); ++x) {
                imgRGBA.SetPixelColor(x, y, img.PixelColor(x, y));
            }
        }

        // Post-process to make background transparent
        for (Standard_Size y = 0; y < imgRGBA.Height(); ++y) {
            for (Standard_Size x = 0; x < imgRGBA.Width(); ++x) {
                Quantity_ColorRGBA color = imgRGBA.PixelColor(x, y);
                // Check if pixel matches background color with tolerance
                if (color.GetRGB().SquareDistance(bgColor) < 0.01) {
                    color.SetAlpha(0.0);
                } else {
                    color.SetAlpha(1.0);
                }
                imgRGBA.SetPixelColor(x, y, color);
            }
        }
        
        // Save using stbi_write_png to ensure alpha channel is preserved
        int stride = (int)imgRGBA.SizeRowBytes();
        if (stbi_write_png(args.output_path.c_str(), (int)imgRGBA.Width(), (int)imgRGBA.Height(), 4, imgRGBA.Data(), stride)) {
             std::cout << "Rendered: " << args.output_path << std::endl;
        } else {
             std::cerr << "Failed to save image: " << args.output_path << std::endl;
        }
    } else {
        std::cerr << "Failed to render image." << std::endl;
    }

    [pool release];
    return 0;
}