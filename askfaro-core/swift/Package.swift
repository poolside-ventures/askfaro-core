// swift-tools-version: 6.2
// Vendored Apple Foundation Models bridge for askfaro-core-generation. Built and
// linked by build.rs via swift-rs (SwiftLinker) only when the `apple-fm` feature
// is enabled on an Apple target. Produces a static library `AppleFM` exposing two
// @_cdecl entry points (afm_availability, afm_generate) consumed from Rust.
import PackageDescription

let package = Package(
    name: "AppleFM",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        .library(name: "AppleFM", type: .static, targets: ["AppleFM"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Brendonovich/swift-rs", from: "1.0.7"),
    ],
    targets: [
        .target(
            name: "AppleFM",
            dependencies: [
                .product(name: "SwiftRs", package: "swift-rs"),
            ],
            swiftSettings: [.swiftLanguageMode(.v6)]
        ),
    ]
)
