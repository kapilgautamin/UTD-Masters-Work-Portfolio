#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>	
#include <glm\gtx\string_cast.hpp>
#include <GL\glew.h>

class Cloth_Animation
{
public:
	Cloth_Animation();
	~Cloth_Animation();

	void init();
	void update(float delta_time);
	void reset();

	void init_cloth();
	void build_spring(int a, int b, float ks, float kd, int type);
	void springify_mesh();
	void add_forces();
	void explicit_euler(float deltaTime);
	void cloth_sphere_collision();
	bool force_applicable(int point);

	struct Spring {
		int p1, p2;
		float rest_length;
		float Ks, Kd;
		int type;
	};
public:
	std::vector<glm::vec4> colors;

	std::vector<glm::vec3> position_vec;
	std::vector<glm::vec3> velocity_vec;
	std::vector<glm::vec3> force_vec;

	std::vector<glm::vec3> triangles;
	std::vector<glm::vec3> points;

	glm::mat4 sphere;
	glm::mat4 inverse_sphere;
	glm::vec3 sphere_center;

	bool enable_gravity = true;
	bool anti_gravity = false;
	bool spring_forces = true;
	bool enable_wind = false;

	bool show_points = false;
	bool fill_cloth = true;

	bool release_corner_1 = false;
	bool release_corner_2 = false;
	bool release_corner_3 = false;
	bool release_corner_4 = false;

	glm::vec3 wind_position;
	glm::vec3 sphere_position;

	int numX = 9, numY = 9;
	const size_t total_points = (numX + 1) * (numY + 1);
	float fullsize = 4.0f;
	float halfsize = fullsize / 2.0f;

	int first_corner = 0;
	int second_corner = 9;
	int third_corner = 90;
	int fourth_corner = 99;

	std::vector<GLushort> indices;
	std::vector<Spring> springs;

	const int EDGE_SPRING = 0;
	const int DIAGONAL_SPRING = 1;
	const int ANGLE_SPRING = 2;
	int spring_count = 0;

	float KsEdge = 0.75f, KdEdge = -0.25f;
	float KsDiagonal = 0.75f, KdDiagonal = -0.25f;
	float KsAngle = 0.95f, KdAngle = -0.25f;
	glm::vec3 gravity = glm::vec3(0.0f, -0.0098f, 0.0f);
	glm::vec3 wind = glm::vec3(0, 0, 0);

	float mass = 0.2f;
	float windStrength = 1.0f;
	float gravityStrength = 1.0f;

	GLdouble MV[16];
	GLdouble P[16];

	glm::vec3 Up = glm::vec3(0, 1, 0), Right, viewDir;
};

