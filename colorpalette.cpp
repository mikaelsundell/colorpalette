//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2022 - present Mikael Sundell.
//

#include <iostream>
#include <vector>
#include <random>

// openimageio
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/typedesc.h>
#include <OpenImageIO/argparse.h>
#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/sysutil.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/imagebufalgo.h>

// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

using namespace OIIO;

// prints
template <typename T>
static void
print_info(std::string param, const T& value = T()) {
    std::cout << "info: " << param << value << std::endl;
}

static void
print_info(std::string param) {
    print_info<std::string>(param);
}

template <typename T>
static void
print_warning(std::string param, const T& value = T()) {
    std::cout << "warning: " << param << value << std::endl;
}

static void
print_warning(std::string param) {
    print_warning<std::string>(param);
}

template <typename T>
static void
print_error(std::string param, const T& value = T()) {
    std::cerr << "error: " << param << value << std::endl;
}

static void
print_error(std::string param) {
    print_error<std::string>(param);
}

// colorpalette tool
struct ColorPaletteTool
{
    bool help = false;
    bool verbose = false;
    bool debug = false;
    std::string inputfile;
    std::string outputfile;
    std::string posterfile;
    int colors = 8;
    int code = EXIT_SUCCESS;
};

static ColorPaletteTool tool;

static int
set_colors(int argc, const char* argv[])
{
    OIIO_DASSERT(argc == 2);
    tool.colors = Strutil::stoi(argv[1]);
    return 0;
}

static int
set_inputfile(int argc, const char* argv[])
{
    OIIO_DASSERT(argc == 2);
    tool.inputfile = argv[1];
    return 0;
}

static int
set_outputfile(int argc, const char* argv[])
{
    OIIO_DASSERT(argc == 2);
    tool.outputfile = argv[1];
    return 0;
}

static int
set_posterfile(int argc, const char* argv[])
{
    OIIO_DASSERT(argc == 2);
    tool.posterfile = argv[1];
    return 0;
}

static void
print_help(ArgParse& ap)
{
    ap.print_help();
}

// utils - colors
float bgr_from_hsv(const cv::Vec3f& color)
{
    cv::Mat bgr(1, 1, CV_32FC3, color);
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Vec3f hsvcolor = hsv.at<cv::Vec3f>(0, 0);
    return hsvcolor[0];
}

// colorset
struct Colorset
{
    cv::Vec3f color;
    cv::Vec2f pixel;
    float hue;
    Colorset(const cv::Vec3f& c, const cv::Vec2f& p, float h)
    : color(c), pixel(p), hue(h) {}
};

void renderBoxByColorset(ImageBuf& imagebuf, const Colorset& colorset, int box, int width)
{
    int xbegin = static_cast<int>(colorset.pixel[0]) - box / 2 - width;
    int xend = xbegin + box + 2 * width;
    int ybegin = static_cast<int>(colorset.pixel[1]) - box / 2 - width;
    int yend = ybegin + box + 2 * width;

    float white[3] = {1.0, 1.0, 1.0};
    float color[3] = {
        colorset.color[0],
        colorset.color[1],
        colorset.color[2]
    };
    
    ROI outer(xbegin, xend, ybegin, yend, 0, 1, 0, 3);
    ImageBufAlgo::fill(imagebuf, white, outer);

    int inner_xbegin = xbegin + width;
    int inner_xend = xend - width;
    int inner_ybegin = ybegin + width;
    int inner_yend = yend - width;

    ROI inner(inner_xbegin, inner_xend, inner_ybegin, inner_yend, 0, 1, 0, 3);
    ImageBufAlgo::fill(imagebuf, color, inner);
}

// main
int 
main( int argc, const char * argv[])
{
    // Helpful for debugging to make sure that any crashes dump a stack
    // trace.
    Sysutil::setup_crash_stacktrace("stdout");

    Filesystem::convert_native_arguments(argc, (const char**)argv);
    ArgParse ap;

    ap.intro("colorpalette -- a tool to process and create palettes of unique colors from images\n");
    ap.usage("colorpalette [options] filename...")
      .add_help(false)
      .exit_on_error(true);
    
    ap.separator("General flags:");
    ap.arg("--help", &tool.help)
      .help("Print help message");
    
    ap.arg("-v", &tool.verbose)
      .help("Verbose status messages");
    
    ap.arg("--colors %d:COLORS")
      .help("Number of unique colors")
      .action(set_colors);
    
    ap.separator("Input flags:");
    ap.arg("--inputfilename %s:INFILENAME")
      .help("Input filename of image")
      .action(set_inputfile);
    
    ap.separator("Output flags:");
    ap.arg("--outputfilename %s:OUTFILENAME")
      .help("Output filename of palette image")
      .action(set_outputfile);

    ap.arg("--posterfilename %s:POSTERFILENAME")
      .help("Output filename of poster image")
      .action(set_posterfile);
    
    // clang-format on
    if (ap.parse_args(argc, (const char**)argv) < 0) {
        print_error("Could no parse arguments: ", ap.geterror());
        print_help(ap);
        ap.abort();
        return EXIT_FAILURE;
    }
    if (ap["help"].get<int>()) {
        print_help(ap);
        ap.abort();
        return EXIT_SUCCESS;
    }
    if (!tool.inputfile.size()) {
        print_error("missing parameter: ", "inputfilename");
        ap.briefusage();
        ap.abort();
        return EXIT_FAILURE;
    }
    if (!tool.outputfile.size()) {
        print_error("missing parameter: ", "outputfilename");
        ap.briefusage();
        ap.abort();
        return EXIT_FAILURE;
    }
    if (argc <= 1) {
        ap.briefusage();
        print_error("For detailed help: colorpalette --help");
        return EXIT_FAILURE;
    }
    
    // colorpalette program
    print_info("colorpalette -- is a tool to process and create palettes of unique colors from images");
    
    // image
    {
        print_info("Reading input file: ", tool.inputfile);
        ImageBuf imagebuf = ImageBuf(tool.inputfile);
        {
            if (!imagebuf.read(0, 0, TypeDesc::FLOAT)) {
                std::cerr << "Could not open image: " << tool.inputfile << std::endl;
                return EXIT_FAILURE;
            }

            ImageSpec& spec = imagebuf.specmod();
            int width = spec.width;
            int height = spec.height;
            int nchannels = spec.nchannels;
            
            // chart file
            if (tool.verbose) {
                print_info("Image attributes: ", "raw");
                print_info("  width: ", width);
                print_info("  height: ", height);
                print_info("  channels: ", nchannels);
                print_info("  metadata: ", "raw");
                for (size_t i = 0; i < spec.extra_attribs.size(); ++i) {
                    const ParamValue& attrib = spec.extra_attribs[i];
                    print_info(" name: ", attrib.name());
                    print_info("   type: ", attrib.type());
                    print_info("   value: " , attrib.get_string());
                }
            }
            
            // opencv
            {
                if (tool.debug) {
                    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_DEBUG);
                } else {
                    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
                }
                
                // image
                cv::Mat image(height, width, CV_32FC3);
                ROI roi = ROI(0, imagebuf.spec().width, 0, imagebuf.spec().height, 0, 1, 0, 3); // copy RGB
                if (!imagebuf.get_pixels(roi, OIIO::TypeDesc::FLOAT, image.data)) {
                    print_error("could not copy pixels from image buffer to opencv for: ", tool.inputfile);
                    return EXIT_FAILURE;
                }

                // Prepare the image data for k-means clustering
                // The image is reshaped into a 2D matrix where each row represents a pixel and each column represents a color channel.
                // This serialization flattens the image into a single row for processing with k-means.
                cv::Mat serialized = image.reshape(1, image.total());
                serialized.convertTo(serialized, CV_32F);

                // Perform k-Means clustering
                // k-means is applied to find 'k' clusters within the color space of the image.
                // 'k' is predefined, aiming for a broad initial capture of the image's color diversity.
                int k = 20;
                std::vector<int> labels;
                cv::Mat centers;
                cv::kmeans(serialized, k, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

                // Diversity selection logic
                // Calculates pairwise distances between all cluster centers to identify similar colors.
                std::vector<std::vector<double>> distances(centers.rows, std::vector<double>(centers.rows, 0));
                for (int i = 0; i < centers.rows; ++i) {
                    for (int j = i + 1; j < centers.rows; ++j) {
                        distances[i][j] = distances[j][i] = cv::norm(centers.row(i) - centers.row(j));
                    }
                }
                
                // Select a diverse set of colors.
                // Aim to select 'tool.colors' that are as distinct as possible based on their pairwise distances.
                std::set<int> selectedindices;
                while (selectedindices.size() < tool.colors)
                {
                    double maxmindistance = 0;
                    int candidateindex = -1;
                    for (int i = 0; i < centers.rows; ++i)
                        {
                        if (selectedindices.find(i) != selectedindices.end()) continue;
                        double minDistance = std::numeric_limits<double>::max();
                        for (int j : selectedindices)
                        {
                            minDistance = std::min(minDistance, distances[i][j]);
                        }
                        if (minDistance > maxmindistance)
                        {
                            maxmindistance = minDistance;
                            candidateindex = i;
                        }
                    }
                    if (candidateindex != -1) {
                        selectedindices.insert(candidateindex);
                    } else {
                        break;
                    }
                }
                
                // Create a matrix for the selected diverse centers.
                cv::Mat diversecenters(selectedindices.size(), centers.cols, centers.type());
                int idx = 0;
                for (int selectedIndex : selectedindices) {
                    centers.row(selectedIndex).copyTo(diversecenters.row(idx++));
                }

                // Reassign each pixel in the image to the color of the closest diverse center.
                // This step alters the original serialized image data to reflect the reduced color palette.
                for (int i = 0; i < labels.size(); ++i) {
                    int clusterIndex = std::distance(selectedindices.begin(), selectedindices.find(labels[i]));
                    for (int j = 0; j < 3; ++j) { // assuming 3 channels
                        serialized.at<float>(i * 3 + j) = diversecenters.at<float>(clusterIndex, j);
                    }
                }

                // Convert back to the original format and save the "poster"
                if (tool.posterfile.length() > 0)
                {
                    cv::Mat clusteredimage = serialized.reshape(3, image.rows);
                    clusteredimage.convertTo(clusteredimage, CV_8UC3, 255.0);
                    cv::Mat correctedimage;
                    cv::cvtColor(clusteredimage, correctedimage, cv::COLOR_RGB2BGR);

                    bool success = cv::imwrite(tool.posterfile, correctedimage);
                    if (!success) {
                        print_error("failed to create clustered image for: ", tool.posterfile);
                        EXIT_FAILURE;
                    }
                }
                    
                print_info("Writing color and pixel values for color palette");
                std::vector<Colorset> colorsets;
                const int seed = 101010; // ultimate question of life in binary
                std::mt19937 gen(seed);
                
                // Convert to hue and get pixel (x, y) coordinates
                std::unordered_map<int, std::vector<int>> indicesmap;
                for (int i = 0; i < labels.size(); ++i) {
                    indicesmap[labels[i]].push_back(i);
                }
                
                for (int i = 0; i < diversecenters.rows; ++i) {
                    cv::Vec3f center = diversecenters.at<cv::Vec3f>(i);
                    float hue = bgr_from_hsv(center);

                    auto it = std::next(selectedindices.begin(), i);
                    if (it != selectedindices.end()) {
                        int originallabel = *it;
                        std::vector<int> &indices = indicesmap[originallabel];

                        // randomly select an index from the vector for this label
                        std::uniform_int_distribution<> dis(0, indices.size() - 1);
                        int pixelindex = indices[dis(gen)];
                        int pixelx = pixelindex % width;
                        int pixely = pixelindex / width;

                        colorsets.emplace_back(center, cv::Vec2f(pixelx, pixely), hue);
                    }
                }
                
                std::sort(colorsets.begin(), colorsets.end(), [](const Colorset& a, const Colorset& b) {
                    return a.hue < b.hue;
                });
                
                // write output image
                print_info("Writing output image with color palette: ", tool.outputfile);
                {
                    int height = 100;
                    int box = imagebuf.spec().width / colorsets.size();

                    ImageSpec spec(
                        imagebuf.spec().width,
                        imagebuf.spec().height + height,
                        3,
                        TypeDesc::FLOAT
                    );
                    ImageBuf copybuf(spec);

                    // copy the original image to imagebuf
                    ImageBufAlgo::paste(copybuf, 0, 0, 0, 0, imagebuf);

                    // fill in the color boxes
                    for (int i = 0; i < colorsets.size(); ++i) {
                        const auto& colorset = colorsets[i];
                        std::ostringstream stream;
                        stream << i + 1
                               << ": "
                               << " rgb: " << colorset.color[0] * 255 << ", " << colorset.color[1] * 255 << ", " << colorset.color[2] * 255
                               << " xy: " << colorset.pixel[0] << ", " << colorset.pixel[1];

                        print_info("Color: ", stream.str());

                        // render palette
                        {
                            float color[3] = {
                                colorsets[i].color[0],
                                colorsets[i].color[1],
                                colorsets[i].color[2]
                            };
                            ROI roi(
                                i * box, (i + 1) * box,
                                imagebuf.spec().height,
                                imagebuf.spec().height + height
                            );
                            ImageBufAlgo::fill(copybuf, color, roi);
                        }
                        
                        // render boxes
                        {
                            int size = 40;
                            int border = 1;
                            float white[3] = {1.0, 1.0, 1.0};
                            renderBoxByColorset(copybuf, colorsets[i], size, border);
                        }
                    }
                    imagebuf.copy(copybuf);
                    
                    if (!imagebuf.write(tool.outputfile)) {
                        print_error("could not write output file", imagebuf.geterror());
                        return EXIT_FAILURE;
                    }
                }
            }
        }
    }
    return 0;
}
