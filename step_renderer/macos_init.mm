#import <Cocoa/Cocoa.h>

extern "C" void macos_init_app() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    }
}
