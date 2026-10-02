#!/usr/bin/env swift
// ocr_screenshots.swift — Extract text from a folder of PNG screenshots using macOS Vision framework.
// Usage: swift ocr_screenshots.swift /path/to/folder
// Output: prints "### filename" headers followed by recognized text, to stdout.

import Vision
import Foundation
import AppKit

let args = CommandLine.arguments
guard args.count >= 2 else {
    fputs("Usage: swift ocr_screenshots.swift /path/to/folder\n", stderr)
    exit(1)
}

let folder = args[1]
let fm = FileManager.default

guard fm.fileExists(atPath: folder) else {
    fputs("Error: folder not found: \(folder)\n", stderr)
    exit(1)
}

let files: [String]
do {
    files = try fm.contentsOfDirectory(atPath: folder)
        .filter { $0.hasSuffix(".png") || $0.hasSuffix(".jpg") || $0.hasSuffix(".jpeg") }
        .sorted()
} catch {
    fputs("Error reading directory: \(error)\n", stderr)
    exit(1)
}

if files.isEmpty {
    fputs("No PNG/JPG images found in: \(folder)\n", stderr)
    exit(1)
}

for file in files {
    let path = "\(folder)/\(file)"
    let url = URL(fileURLWithPath: path)
    guard let image = NSImage(contentsOf: url),
          let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("### \(file): FAILED TO LOAD")
        continue
    }
    print("### \(file)")
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    let handler = VNImageRequestHandler(cgImage: cgImage)
    try! handler.perform([request])
    let observations = request.results ?? []
    for obs in observations {
        if let candidate = obs.topCandidates(1).first {
            print(candidate.string)
        }
    }
    print("")
}
