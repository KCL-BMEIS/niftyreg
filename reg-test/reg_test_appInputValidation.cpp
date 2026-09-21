#include "reg_test_common.h"
#include "_reg_ReadWriteImage.h"
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#ifndef _WIN32
#include <sys/wait.h>
#endif

/**
 *  Input validation of the command-line tools on mixed 2D/3D inputs
 *
 *  The tools dimension their buffers from one input - the reference image, or the first
 *  transformation - and evaluate every other input into them, so a 2D input paired with a 3D
 *  one has to be refused before anything is allocated. That refusal lives in each tool's main
 *  function and is only observable from outside: the tools are run here as processes on tiny
 *  images in a temporary directory and must end with a failure status - not a signal - after
 *  naming the offending pair. A matched run per tool checks that valid inputs are accepted.
 */

using Catch::Matchers::ContainsSubstring;

namespace {

// Where CMake placed the tools
const std::filesystem::path kAppDir = NR_APP_DIR;

std::string App(const std::string& name) {
#ifdef _WIN32
    return (kAppDir / (name + ".exe")).string();
#else
    return (kAppDir / name).string();
#endif
}

// A temporary directory that goes away with the test case, named uniquely across concurrent
// test processes
struct ScratchDir {
    std::filesystem::path path;
    ScratchDir() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        const auto salt = std::random_device{}();
        path = std::filesystem::temp_directory_path() /
               ("niftyreg_app_validation_" + std::to_string(stamp) + "_" + std::to_string(salt));
        std::filesystem::create_directories(path);
    }
    ~ScratchDir() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
    std::string File(const std::string& name) const { return (path / name).string(); }
};

struct RunResult {
    int exitCode = -1;
    bool crashed = false;   // ended by a signal rather than by returning from main
    std::string output;     // stdout and stderr
};

std::string Quote(const std::string& arg) { return "\"" + arg + "\""; }

// Run a tool with the given arguments, capturing its output in the scratch directory
RunResult Run(const ScratchDir& scratch, const std::string& app, const vector<std::string>& args) {
    const std::string logFile = scratch.File("output.log");
    std::string command = Quote(App(app));
    for (const auto& arg : args)
        command += " " + Quote(arg);
    command += " > " + Quote(logFile) + " 2>&1";
#ifdef _WIN32
    // cmd.exe strips the first and last quote of a command that starts with one; an outer pair
    // absorbs that
    command = "\"" + command + "\"";
#endif
    RunResult result;
    const int status = std::system(command.c_str());
#ifdef _WIN32
    result.exitCode = status;
#else
    if (status != -1 && WIFEXITED(status))
        result.exitCode = WEXITSTATUS(status);
    else
        result.crashed = true;
#endif
    std::ifstream log(logFile);
    std::stringstream buffer;
    buffer << log.rdbuf();
    result.output = buffer.str();
    return result;
}

// Smooth, strictly positive intensities so that every similarity measure has something to work on
NiftiImage MakeImage(const vector<NiftiImage::dim_t>& dims, const float offset) {
    NiftiImage image(dims, NIFTI_TYPE_FLOAT32);
    mat44 sform;
    Mat44Eye(&sform);
    for (int i = 0; i < 3; ++i) sform.m[i][3] = offset;
    image->sform_code = 1;
    image->sto_xyz = sform;
    image->sto_ijk = nifti_mat44_inverse(sform);
    image->qform_code = 0;
    auto data = image.data();
    size_t index = 0;
    for (int z = 0; z < image->nz; ++z)
        for (int y = 0; y < image->ny; ++y)
            for (int x = 0; x < image->nx; ++x, ++index)
                data[index] = 100.f + 40.f * cosf(0.31f * x + 0.4f) * cosf(0.23f * y + 1.1f) * cosf(0.27f * z + 0.7f)
                            + 25.f * cosf(0.11f * x - 0.07f * y + 0.19f * z + 2.f);
    return image;
}

NiftiImage MakeMask(const vector<NiftiImage::dim_t>& dims) {
    NiftiImage mask(dims, NIFTI_TYPE_UINT8);
    auto data = mask.data();
    for (size_t i = 0; i < mask.nVoxels(); ++i)
        data[i] = 1;
    return mask;
}

void Write(NiftiImage& image, const std::string& file) {
    reg_io_WriteImageFile(image, file.c_str());
}

void WriteIdentityAffine(const std::string& file) {
    std::ofstream affine(file);
    affine << "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n";
}

// Every input the cases below need, in both dimensionalities, on disk
struct Inputs {
    ScratchDir scratch;
    std::string ref2d, ref3d, flo2d, flo3d, mask2d, mask3d, def2d, def3d, cpp2d, cpp3d, affine;
    Inputs() {
        NiftiImage image2d = MakeImage({ 16, 16 }, 0.f);
        NiftiImage image3d = MakeImage({ 12, 12, 12 }, 0.f);
        NiftiImage floating2d = MakeImage({ 16, 16 }, 1.f);
        NiftiImage floating3d = MakeImage({ 12, 12, 12 }, 1.f);
        NiftiImage maskImage2d = MakeMask({ 16, 16 });
        NiftiImage maskImage3d = MakeMask({ 12, 12, 12 });
        NiftiImage field2d = CreateDeformationField(image2d);
        NiftiImage field3d = CreateDeformationField(image3d);
        NiftiImage grid2d = CreateControlPointGrid(image2d, kProductionGridSpacing);
        NiftiImage grid3d = CreateControlPointGrid(image3d, kProductionGridSpacing);
        Write(image2d, ref2d = scratch.File("ref2d.nii"));
        Write(image3d, ref3d = scratch.File("ref3d.nii"));
        Write(floating2d, flo2d = scratch.File("flo2d.nii"));
        Write(floating3d, flo3d = scratch.File("flo3d.nii"));
        Write(maskImage2d, mask2d = scratch.File("mask2d.nii"));
        Write(maskImage3d, mask3d = scratch.File("mask3d.nii"));
        Write(field2d, def2d = scratch.File("def2d.nii"));
        Write(field3d, def3d = scratch.File("def3d.nii"));
        Write(grid2d, cpp2d = scratch.File("cpp2d.nii"));
        Write(grid3d, cpp3d = scratch.File("cpp3d.nii"));
        WriteIdentityAffine(affine = scratch.File("affine.txt"));
    }
    std::string Out(const std::string& name) const { return scratch.File(name); }
};

// A rejected pairing: the tool returns a failure status and names the disagreement
void RequireRejected(const RunResult& result, const std::string& diagnosis) {
    INFO(result.output);
    REQUIRE_FALSE(result.crashed);
    REQUIRE(result.exitCode == EXIT_FAILURE);
    REQUIRE_THAT(result.output, ContainsSubstring(diagnosis));
}

// A valid run: the tool returns success and produced its output file
void RequireCompleted(const RunResult& result, const std::string& outputFile) {
    INFO(result.output);
    REQUIRE_FALSE(result.crashed);
    REQUIRE(result.exitCode == EXIT_SUCCESS);
    REQUIRE(std::filesystem::exists(outputFile));
}

const std::string kMismatch = "must both be 2D or both be 3D";

} // namespace

TEST_CASE("reg_resample refuses mixed 2D/3D inputs", "[apps][validation]") {
    const Inputs in;
    const std::string out = in.Out("resampled.nii");

    SECTION("Reference and floating images") {
        RequireRejected(Run(in.scratch, "reg_resample", { "-ref", in.ref2d, "-flo", in.flo3d, "-res", out, "-voff" }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_resample", { "-ref", in.ref3d, "-flo", in.flo2d, "-res", out, "-voff" }), kMismatch);
    }
    SECTION("Reference image and transformation") {
        RequireRejected(Run(in.scratch, "reg_resample", { "-ref", in.ref3d, "-flo", in.flo3d, "-trans", in.def2d, "-res", out, "-voff" }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_resample", { "-ref", in.ref2d, "-flo", in.flo2d, "-trans", in.cpp3d, "-res", out, "-voff" }), kMismatch);
    }
    SECTION("Matched inputs") {
        RequireCompleted(Run(in.scratch, "reg_resample", { "-ref", in.ref3d, "-flo", in.flo3d, "-trans", in.cpp3d, "-res", out, "-voff" }), out);
        RequireCompleted(Run(in.scratch, "reg_resample", { "-ref", in.ref2d, "-flo", in.flo2d, "-trans", in.def2d, "-res", out, "-voff" }), out);
    }
}

TEST_CASE("reg_measure refuses mixed 2D/3D inputs", "[apps][validation]") {
    const Inputs in;
    const std::string out = in.Out("measure.txt");

    SECTION("Reference and floating images") {
        RequireRejected(Run(in.scratch, "reg_measure", { "-ref", in.ref3d, "-flo", in.flo2d, "-ncc" }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_measure", { "-ref", in.ref2d, "-flo", in.flo3d, "-nmi" }), kMismatch);
    }
    SECTION("Reference image and mask") {
        RequireRejected(Run(in.scratch, "reg_measure", { "-ref", in.ref3d, "-flo", in.flo3d, "-rmask", in.mask2d, "-ncc" }),
                        "must be defined on the reference image grid");
    }
    SECTION("Matched inputs") {
        RequireCompleted(Run(in.scratch, "reg_measure", { "-ref", in.ref2d, "-flo", in.flo2d, "-rmask", in.mask2d, "-ncc", "-nmi", "-ssd", "-lncc", "-out", out }), out);
    }
}

TEST_CASE("reg_transform refuses mixed 2D/3D inputs", "[apps][validation]") {
    const Inputs in;
    const std::string out = in.Out("transform.nii");

    SECTION("Deformation field from a grid of the other dimensionality") {
        RequireRejected(Run(in.scratch, "reg_transform", { "-ref", in.ref2d, "-def", in.cpp3d, out }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_transform", { "-ref", in.ref3d, "-disp", in.cpp2d, out }), kMismatch);
    }
    SECTION("Composition of fields of different dimensionality") {
        RequireRejected(Run(in.scratch, "reg_transform", { "-comp", in.def3d, in.def2d, out }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_transform", { "-comp", in.def2d, in.def3d, out }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_transform", { "-ref", in.ref3d, "-comp", in.cpp3d, in.def2d, out }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_transform", { "-ref", in.ref2d, "-comp", in.affine, in.def3d, out }), kMismatch);
    }
    SECTION("Inversion onto a floating image of the other dimensionality") {
        RequireRejected(Run(in.scratch, "reg_transform", { "-invNrr", in.def3d, in.ref2d, out }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_transform", { "-ref", in.ref2d, "-invNrr", in.cpp3d, in.ref3d, out }), kMismatch);
    }
    SECTION("Matched inputs") {
        RequireCompleted(Run(in.scratch, "reg_transform", { "-ref", in.ref3d, "-def", in.cpp3d, out }), out);
        RequireCompleted(Run(in.scratch, "reg_transform", { "-ref", in.ref2d, "-comp", in.cpp2d, in.def2d, out }), out);
        RequireCompleted(Run(in.scratch, "reg_transform", { "-comp", in.def3d, in.def3d, out }), out);
    }
}

TEST_CASE("reg_jacobian refuses mixed 2D/3D inputs", "[apps][validation]") {
    const Inputs in;
    const std::string out = in.Out("jacobian.nii");

    SECTION("Reference image and grid") {
        RequireRejected(Run(in.scratch, "reg_jacobian", { "-ref", in.ref2d, "-trans", in.cpp3d, "-jac", out }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_jacobian", { "-ref", in.ref3d, "-trans", in.cpp2d, "-jacM", out }), kMismatch);
    }
    SECTION("Matched inputs") {
        RequireCompleted(Run(in.scratch, "reg_jacobian", { "-ref", in.ref3d, "-trans", in.cpp3d, "-jac", out }), out);
        RequireCompleted(Run(in.scratch, "reg_jacobian", { "-trans", in.def2d, "-jacM", out }), out);
    }
}

TEST_CASE("reg_tools refuses operands of another shape", "[apps][validation]") {
    const Inputs in;
    const std::string out = in.Out("tools.nii");

    SECTION("Voxel-wise arithmetic") {
        RequireRejected(Run(in.scratch, "reg_tools", { "-in", in.ref3d, "-add", in.flo2d, "-out", out }), "must have the same dimensions");
        RequireRejected(Run(in.scratch, "reg_tools", { "-in", in.ref2d, "-mul", in.flo3d, "-out", out }), "must have the same dimensions");
    }
    SECTION("NaN masking") {
        RequireRejected(Run(in.scratch, "reg_tools", { "-in", in.ref2d, "-nan", in.mask3d, "-out", out }), "must have the same dimensions");
    }
    SECTION("Matched inputs") {
        RequireCompleted(Run(in.scratch, "reg_tools", { "-in", in.ref3d, "-add", in.flo3d, "-out", out }), out);
        RequireCompleted(Run(in.scratch, "reg_tools", { "-in", in.ref2d, "-nan", in.mask2d, "-out", out }), out);
    }
}

TEST_CASE("reg_average refuses mixed 2D/3D inputs", "[apps][validation]") {
    const Inputs in;
    const std::string out = in.Out("average.nii");

    SECTION("Transformation of the other dimensionality") {
        RequireRejected(Run(in.scratch, "reg_average", { out, "-avg_tran", in.ref3d, in.def3d, in.flo3d, in.def2d, in.flo3d }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_average", { out, "-avg_tran", in.ref2d, in.cpp2d, in.flo2d, in.cpp3d, in.flo2d }), kMismatch);
    }
    SECTION("Image of the other dimensionality") {
        RequireRejected(Run(in.scratch, "reg_average", { out, "-avg_tran", in.ref3d, in.affine, in.flo3d, in.affine, in.flo2d }), kMismatch);
        RequireRejected(Run(in.scratch, "reg_average", { out, "-avg", in.ref3d, in.flo2d }), kMismatch);
    }
    SECTION("Matched inputs") {
        RequireCompleted(Run(in.scratch, "reg_average", { out, "-avg_tran", in.ref3d, in.cpp3d, in.flo3d, in.def3d, in.flo3d }), out);
        RequireCompleted(Run(in.scratch, "reg_average", { out, "-avg", in.ref2d, in.flo2d }), out);
    }
}
