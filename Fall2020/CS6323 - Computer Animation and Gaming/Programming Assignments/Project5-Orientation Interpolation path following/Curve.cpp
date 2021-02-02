#include "Curve.h"

Curve::Curve()
{
}

Curve::~Curve()
{
}

void Curve::init()
{
	this->control_points_pos = {
		{ 0.0, 8.5, -2.0 },
		{ -3.0, 11.0, 2.3 },
		{ -6.0, 8.5, -2.5 },
		{ -4.0, 5.5, 2.8 },
		{ 1.0, 2.0, -4.0 },
		{ 4.0, 2.0, 3.0 },
		{ 7.0, 8.0, -2.0 },
		{ 3.0, 10.0, 3.7 }
	};
	calculate_curve();

	this->control_points_quaternion = {
		{0.13964   , 0.0481732 , 0.831429 , 0.541043 , },
		{0.0509038 , -0.033869 , -0.579695, 0.811295 , },
		{-0.502889 , -0.366766 , 0.493961 , 0.592445 , },
		{-0.636    , 0.667177  , -0.175206, 0.198922 , },
		{0.693492  , 0.688833  , -0.152595, -0.108237, },
		{0.752155  , -0.519591 , -0.316988, 0.168866 , },
		{0.542054  , 0.382705  , 0.378416 , 0.646269 , },
		{0.00417342, -0.0208652, -0.584026, 0.810619   }
	};
}

glm::mat4 Curve::quat_to_rotation_mat(glm::quat input) {
	double s = input.w;
	double x = input.x;
	double y = input.y;
	double z = input.z;
	//std::cout << "quat " << input.w << " " << input.x << " " << input.y << " " << input.z << std::endl;
	//std::cout<< "cuat " << s << " " << x  << " " << y  << " " << z  << std::endl;

	glm::mat4 matrix = glm::mat4(1.0f);

	/*matrix[0][0] = 1 - 2 * (y * y + z * z);
	matrix[0][1] = 2 * (x * y -  s * z);
	matrix[0][2] = 2 * (x * z + s * y);
	
	matrix[1][0] = 2 * (x * y + s * z);
	matrix[1][1] = 1 - 2 * (x * x + z * z);
	matrix[1][2] = 2 * (y * z - s * x);
	
	matrix[2][0] = 2 * (x * z - s * y);
	matrix[2][1] = 2 * (y * z + s * x);
	matrix[2][2] = 1 - 2 * (x * x + y * y);*/

	matrix[0][0] = 1 - 2 * (y * y + z * z);
	matrix[0][1] = 2 * (x * y + s * z);
	matrix[0][2] = 2 * (x * z - s * y);

	matrix[1][0] = 2 * (x * y - s * z);
	matrix[1][1] = 1 - 2 * (x * x + z * z);
	matrix[1][2] = 2 * (y * z + s * x);

	matrix[2][0] = 2 * (x * z + s * y);
	matrix[2][1] = 2 * (y * z - s * x);
	matrix[2][2] = 1 - 2 * (x * x + y * y);

	return matrix;
}

void Curve::calculate_curve()
{
	/*
	since this is a closed curve, I added 1 point in the starting and 2 points at the end to complete catmull rom for all segments
	P7,P0,P1,P2 -> P0 P1 -> segment 0
	P0,P1,P2,P3 -> P1 P2 -> segment 1
	P1,P2,P3,P4 -> P2 P3 -> segment 2
	P2,P3,P4,P5 -> P3 P4 -> segment 3
	P3,P4,P5,P6 -> P4 P5 -> segment 4
	P4,P5,P6,P7 -> P5 P6 -> segment 5
	P5,P6,P7,P0 -> P6 P7 -> segment 6
	P6,P7,P0,P1 -> P7 P0 -> segment 7
	*/
	std::vector<glm::vec3> control_points(this->control_points_pos);
	control_points.insert(control_points.begin(), this->control_points_pos[7]);
	control_points.push_back(this->control_points_pos[0]);
	control_points.push_back(this->control_points_pos[1]);


	glm::vec3 p0, p1, p2, p3;
	float increment = 1.0 / this->num_points_per_segment; //0.005
	//std::cout<< "increment " << increment << std::endl;
	float sum_linear_len = 0.0f;
	int cache_count = 0;
	int control_point_size = control_points.size() - 3;
	
	for (int p = 0; p < control_point_size; p++) {
		p0 = control_points[p];
		p1 = control_points[p + 1];
		p2 = control_points[p + 2];
		p3 = control_points[p + 3];
		this->segment_length[p] = 0.0f;
		//std::cout << "segment " << (p + 1) % control_point_size << "sum " << sum_linear_len << std::endl;
		for (float x = 0; x <= 1; x += increment) {
			glm::vec3 point = catmull_rom(x, p0, p1, p2, p3);
			//float approx_arc_len = glm::sqrt(pow(p3.x - point.x, 2) + pow(p3.y - point.y, 2) + pow(p3.z - point.z, 2));
			float approx_arc_len = glm::sqrt(pow(p0.x - point.x, 2) + pow(p0.y - point.y, 2) + pow(p0.z - point.z, 2));
			sum_linear_len += approx_arc_len;
			this->segment_length[p] += approx_arc_len;
			this->cache[cache_count].point = point;
			this->cache[cache_count].segment_no = (p) % control_point_size;
			this->cache[cache_count].length = sum_linear_len;
			cache_count += 1;
			this->curve_points_pos.push_back(point);
		}
		//std::cout << cache_count << "segment " << (p + 1) % control_point_size << "sum " << sum_linear_len << std::endl;
	}
	this->cache_size = cache_count;
	//for (int c = 0; c < this->cache_size; c++)
	//	this->cache[c].length = this->cache[c].length / this->cache[this->cache_size - 1].length;
}

glm::vec3 Curve::catmull_rom(float x, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3)
{
	glm::mat4 catmull_rom_mat = {
		{-0.5,  1.5, -1.5,  0.5},
		{   1, -2.5,    2, -0.5},
		{-0.5,    0,  0.5,    0},
		{   0,    1,    0,    0}
	};

	float xsquare = x * x;
	float xcube = x * x * x;

	glm::vec4 input = { xcube, xsquare, x, 1 };
	glm::mat4x3 points = { p0,p1,p2,p3 };
	//std::cout << glm::to_string(points * catmull_rom_mat * input) << std::endl;
	//column order of 1x4 * 4x4 * 4x3
	return points * catmull_rom_mat * input;
}
