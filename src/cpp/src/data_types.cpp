#include "../include/data_types.hpp"

namespace {
    constexpr double PI = 3.14159265358979323846;

    inline double deg2rad(double deg) {
        return deg * (PI / 180.0);
    }

    inline double rad2deg(double rad) {
        return rad * (180.0 / PI);
    }
}

// -----------------------------------------------------------------------------
// Point Implementation
// -----------------------------------------------------------------------------

Point::Point(double x_val, double y_val) : x(x_val), y(y_val) {}

std::tuple<double, double> Point::as_tuple() const {
    return {x, y};
}

// -----------------------------------------------------------------------------
// GeographicPoint Implementation
// -----------------------------------------------------------------------------

GeographicPoint::GeographicPoint(double lon_val, double lat_val) 
    : lon(lon_val), lat(lat_val) {
    if (lon < -180.0 || lon > 180.0) {
        throw ValidationError("Longitude must be between -180 and 180 degrees");
    }
    if (lat < -90.0 || lat > 90.0) {
        throw ValidationError("Latitude must be between -90 and 90 degrees");
    }
}

GeographicPoint GeographicPoint::normalize_coordinates() const {
    double normalized_lon = std::fmod(lon + 180.0, 360.0);
    if (normalized_lon < 0) normalized_lon += 360.0;
    normalized_lon -= 180.0;

    double normalized_lat = std::fmod(lat + 90.0, 180.0);
    if (normalized_lat < 0) normalized_lat += 180.0;
    normalized_lat -= 90.0;

    return GeographicPoint(normalized_lon, normalized_lat);
}

std::tuple<double, double> GeographicPoint::as_tuple() const {
    return {lon, lat};
}

// -----------------------------------------------------------------------------
// Vector Implementation
// -----------------------------------------------------------------------------

Vector::Vector(double x_val, double y_val, double z_val) 
    : x(x_val), y(y_val), z(z_val) {
    if (length() == 0.0) {
        throw ValidationError("Vector cannot have zero length");
    }
}

double Vector::length() const {
    return std::sqrt(x * x + y * y + z * z);
}

Vector Vector::normalize() const {
    double len = length();
    if (len == 0.0) {
        throw ValidationError("Cannot normalize zero-length vector");
    }
    return Vector(x / len, y / len, z / len);
}

double Vector::dot(const Vector& other) const {
    return x * other.x + y * other.y + z * other.z;
}

std::tuple<double, double, double> Vector::as_tuple() const {
    return {x, y, z};
}

Vector Vector::round(int decimals, bool round_down) const {
    double factor = std::pow(10.0, decimals);
    
    auto round_val = [factor, round_down](double val) {
        if (round_down) {
            return std::floor(val * factor) / factor;
        } else {
            return std::round(val * factor) / factor;
        }
    };

    return Vector(round_val(x), round_val(y), round_val(z));
}

GeographicPoint Vector::to_spherical() const {
    double norm = std::sqrt(x * x + y * y + z * z);
    
    double _lon = std::round(rad2deg(std::atan2(y, x)) * 1e6) / 1e6;
    double _lat = std::round(rad2deg(std::asin(z / norm)) * 1e6) / 1e6;

    if (_lon > 180.0) {
        _lon -= 360.0;
    } else if (_lon <= -180.0) {
        _lon += 360.0;
    }

    return GeographicPoint(_lon, _lat);
}

Vector Vector::from_string(const std::string& vector_str) {
    static const std::regex pattern(R"(Vector\(x=([\d\.\-eE]+),\s*y=([\d\.\-eE]+),\s*z=([\d\.\-eE]+)\))");
    std::smatch match;

    if (std::regex_match(vector_str, match, pattern)) {
        double x_val = std::stod(match[1].str());
        double y_val = std::stod(match[2].str());
        double z_val = std::stod(match[3].str());
        return Vector(x_val, y_val, z_val);
    } else {
        throw std::invalid_argument("Invalid vector string format: " + vector_str);
    }
}

Vector Vector::from_geographic(double lon, double lat) {
    if (lon < -180.0 || lon > 180.0) {
        throw ValidationError("Longitude must be between -180 and 180 degrees");
    }
    if (lat < -90.0 || lat > 90.0) {
        throw ValidationError("Latitude must be between -90 and 90 degrees");
    }

    double theta = deg2rad(lon);
    double phi = deg2rad(90.0 - lat); // Convert to co-latitude

    double x_val = std::sin(phi) * std::cos(theta);
    double y_val = std::sin(phi) * std::sin(theta);
    double z_val = std::cos(phi);

    return Vector(x_val, y_val, z_val);
}

// -----------------------------------------------------------------------------
// Free Function Implementation
// -----------------------------------------------------------------------------

std::pair<std::vector<double>, std::vector<double>> convert_vectors_to_coordinates(
    const std::vector<Vector>& vectors
) {
    if (vectors.empty()) {
        throw ValidationError("Empty vector collection provided");
    }

    try {
        std::vector<double> lons;
        std::vector<double> lats;
        lons.reserve(vectors.size());
        lats.reserve(vectors.size());

        for (const auto& v : vectors) {
            double norm = v.length();
            double lon = rad2deg(std::atan2(v.y, v.x));
            double lat = rad2deg(std::asin(v.z / norm));

            if (lon > 180.0) {
                lon -= 360.0;
            } else if (lon <= -180.0) {
                lon += 360.0;
            }

            lons.push_back(lon);
            lats.push_back(lat);
        }

        return {lons, lats};
    } catch (const std::exception& e) {
        throw ValidationError(std::string("Failed to convert vectors to coordinates: ") + e.what());
    }
}