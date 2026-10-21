#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <cmath>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

// Custom Exception for validation failures
class ValidationError : public std::runtime_error {
public:
    explicit ValidationError(const std::string& message)
        : std::runtime_error(message) {}
};

// Represents a point in 2D pixel coordinates
struct Point {
    double x{0.0};
    double y{0.0};

    Point(double x_val, double y_val);

    std::tuple<double, double> as_tuple() const;
};

// Represents a point on a sphere using longitude and latitude coordinates (degrees)
struct GeographicPoint {
    double lon{0.0};
    double lat{0.0};

    GeographicPoint(double lon_val, double lat_val);

    GeographicPoint normalize_coordinates() const;
    std::tuple<double, double> as_tuple() const;
};

// Represents a point or unit direction vector in 3D Cartesian coordinates
struct Vector {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    Vector(double x_val, double y_val, double z_val);

    double length() const;
    Vector normalize() const;
    double dot(const Vector& other) const;
    std::tuple<double, double, double> as_tuple() const;
    Vector round(int decimals, bool round_down = false) const;
    GeographicPoint to_spherical() const;

    static Vector from_string(const std::string& vector_str);
    static Vector from_geographic(double lon, double lat);
};

// Utility function to convert a collection of vectors into longitude and latitude arrays
std::pair<std::vector<double>, std::vector<double>> convert_vectors_to_coordinates(
    const std::vector<Vector>& vectors
);

#endif // DATA_TYPES_HPP