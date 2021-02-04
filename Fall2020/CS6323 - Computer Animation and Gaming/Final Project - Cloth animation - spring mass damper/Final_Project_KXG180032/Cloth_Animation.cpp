#include "Cloth_Animation.h"
#include "Object.h"

Cloth_Animation::Cloth_Animation()
{
}

Cloth_Animation::~Cloth_Animation()
{
}

void Cloth_Animation::init()
{
	wind_position = { 2.0f, 0.5f, 2.0f };
	sphere_position = { -0.5f, 2.0f, -0.5f };

	colors =
	{
		{0.7f,0.0f,0.0f,1.0f},
		{0.7f,0.7f,0.0f,1.0f},
		{0.7f,0.0f,0.7f,1.0f},
		{0.0f,0.7f,0.7f,1.0f},
		{0.0f,0.85f,0.0f,1.0f}
	};

}

void Cloth_Animation::init_cloth() {
	int i = 0, j = 0, count = 0;
	int v = numY + 1;
	int u = numX + 1;

	indices.resize(numX * numY * 2 * 3);
	position_vec.resize(total_points);
	velocity_vec.resize(total_points);
	force_vec.resize(total_points);

	//fill in position_vec
	for (j = 0; j < v; j++) {
		for (i = 0; i < u; i++) {
			position_vec[count++] = glm::vec3(((float(i) / (u - 1)) * 2 - 1) * halfsize, fullsize + 1, ((float(j) / (v - 1)) * fullsize));
		}
	}

	//fill in velocity_vec
	memset(&(velocity_vec[0].x), 0, total_points * sizeof(glm::vec3));

	//fill in indices
	GLushort* id = &indices[0];
	for (i = 0; i < numY; i++) {
		for (j = 0; j < numX; j++) {
			int i0 = i * (numX + 1) + j;
			int i1 = i0 + 1;
			int i2 = i0 + (numX + 1);
			int i3 = i2 + 1;
			if ((j + i) % 2) {
				*id++ = i0; *id++ = i2; *id++ = i1;
				*id++ = i1; *id++ = i2; *id++ = i3;
			}
			else {
				*id++ = i0; *id++ = i2; *id++ = i3;
				*id++ = i0; *id++ = i3; *id++ = i1;
			}
		}
	}

	springify_mesh();
}

void Cloth_Animation::build_spring(int a, int b, float ks, float kd, int type) {
	Spring spring;
	spring.p1 = a;
	spring.p2 = b;
	spring.Ks = ks;
	spring.Kd = kd;
	spring.type = type;
	glm::vec3 deltaP = position_vec[a] - position_vec[b];
	spring.rest_length = sqrt(glm::dot(deltaP, deltaP));
	springs.push_back(spring);
}

void Cloth_Animation::springify_mesh()
{
	int l1 = 0, l2 = 0;
	int v = numY + 1;
	int u = numX + 1;

	// Build EDGE_SPRING - Horizontal
	for (l1 = 0; l1 < v; l1++)
		for (l2 = 0; l2 < (u - 1); l2++) {
			build_spring((l1 * u) + l2, (l1 * u) + l2 + 1, KsEdge, KdEdge, EDGE_SPRING);

		}

	// Build EDGE_SPRING - Vertical 
	for (l1 = 0; l1 < (u); l1++)
		for (l2 = 0; l2 < (v - 1); l2++) {
			build_spring((l2 * u) + l1, ((l2 + 1) * u) + l1, KsEdge, KdEdge, EDGE_SPRING);
		}


	// Build DIAGONAL_SPRING - Top to Bottom
	for (l1 = 0; l1 < (v - 1); l1++)
		for (l2 = 0; l2 < (u - 1); l2++) {
			build_spring((l1 * u) + l2, ((l1 + 1) * u) + l2 + 1, KsDiagonal, KdDiagonal, DIAGONAL_SPRING);
		}

	// Build DIAGONAL_SPRING - Bottom to Top
	for (l1 = 0; l1 < (v - 1); l1++)
		for (l2 = 0; l2 < (u - 1); l2++) {
			build_spring(((l1 + 1) * u) + l2, (l1 * u) + l2 + 1, KsDiagonal, KdDiagonal, DIAGONAL_SPRING);
		}

	// Build ANGLE_SPRING - Horizontal
	for (l1 = 0; l1 < (v); l1++) {
		for (l2 = 0; l2 < (u - 2); l2++) {
			build_spring((l1 * u) + l2, (l1 * u) + l2 + 2, KsAngle, KdAngle, ANGLE_SPRING);
		}
		build_spring((l1 * u) + (u - 3), (l1 * u) + (u - 1), KsAngle, KdAngle, ANGLE_SPRING);
	}

	// Build ANGLE_SPRING - Vertical
	for (l1 = 0; l1 < (u); l1++) {
		for (l2 = 0; l2 < (v - 2); l2++) {
			build_spring((l2 * u) + l1, ((l2 + 2) * u) + l1, KsAngle, KdAngle, ANGLE_SPRING);
		}
		build_spring(((v - 3) * u) + l1, ((v - 1) * u) + l1, KsAngle, KdAngle, ANGLE_SPRING);
	}
}

void Cloth_Animation::add_forces() {
	size_t i = 0;
	
	wind = wind_position;

	for (i = 0; i < total_points; i++) {
		force_vec[i] = glm::vec3(0);
		//do not apply gravity or wind to corners
		if (force_applicable(i)) {
			if (enable_gravity)
				force_vec[i] += (gravity * gravityStrength);
			else if (anti_gravity)
				force_vec[i] += (-1 * gravity.y * gravityStrength);

			if (enable_wind)
				force_vec[i] += (wind * windStrength * 0.01f);
		}
	}

	if (spring_forces) {
		//add spring forces
		for (i = 0; i < springs.size(); i++) {
			Spring curr_spring = springs[i];
			int spring_left = curr_spring.p1;
			int spring_right = curr_spring.p2;
			
			glm::vec3 p1 = position_vec[spring_left];
			glm::vec3 p2 = position_vec[spring_right];
			glm::vec3 v1 = velocity_vec[spring_left];
			glm::vec3 v2 = velocity_vec[spring_right];
			glm::vec3 deltaP = p1 - p2;
			glm::vec3 deltaV = v1 - v2;
			float dist = glm::length(deltaP);

			float leftTerm = -springs[i].Ks * (dist - springs[i].rest_length);
			float rightTerm = springs[i].Kd * (glm::dot(deltaV, deltaP) / dist);
			glm::vec3 springForce = (leftTerm + rightTerm) * glm::normalize(deltaP);

			if(force_applicable(spring_left))
				force_vec[spring_left] += springForce;

			if(force_applicable(spring_right))
			force_vec[spring_right] -= springForce;
		}
	}
}

bool Cloth_Animation::force_applicable(int point) {
	if (point == first_corner && release_corner_1)
		return false;
	if (point == second_corner && release_corner_2)
		return false;
	if (point == third_corner && release_corner_3)
		return false;
	if (point == fourth_corner && release_corner_4)
		return false;
	return true;
}

void Cloth_Animation::explicit_euler(float deltaTime) {
	float deltaTimeMass = deltaTime / mass;
	size_t i = 0;

	for (i = 0; i < total_points; i++) {
		glm::vec3 oldV = velocity_vec[i];
		velocity_vec[i] += (force_vec[i] * deltaTimeMass);
		position_vec[i] += deltaTime * oldV;

		if (position_vec[i].y < 0) {
			position_vec[i].y = 0;
		}
	}
}

void Cloth_Animation::cloth_sphere_collision() {

	float radius = 1.0f;

	for (size_t i = 0; i < total_points; i++) {
		glm::vec4 temp = (inverse_sphere * glm::vec4(position_vec[i], 1));
		glm::vec3 delta_distance = glm::vec3(temp.x, temp.y, temp.z) - sphere_center;
		float distance = glm::length(delta_distance);
		if (distance < 1.0f) {
			delta_distance = (radius - distance) * delta_distance / distance;

			glm::vec3 delta;
			glm::vec3 transformInv;
			transformInv = glm::vec3(sphere[0].x, sphere[1].x, sphere[2].x);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.x = glm::dot(delta_distance, transformInv);

			transformInv = glm::vec3(sphere[0].y, sphere[1].y, sphere[2].y);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.y = glm::dot(delta_distance, transformInv);

			transformInv = glm::vec3(sphere[0].z, sphere[1].z, sphere[2].z);
			transformInv /= glm::dot(transformInv, transformInv);
			delta.z = glm::dot(delta_distance, transformInv);

			position_vec[i] += delta;
			velocity_vec[i] = glm::vec3(0);
		}
	}
}

void Cloth_Animation::update(float delta_time)
{
	if ((enable_gravity || anti_gravity) && (release_corner_1 || release_corner_2 || release_corner_3 || release_corner_4)) {
		add_forces();
		explicit_euler(delta_time);
		cloth_sphere_collision();
	}
}

void Cloth_Animation::reset()
{
	position_vec.clear();
	velocity_vec.clear();
	force_vec.clear();
	triangles.clear();
	points.clear();
	init_cloth();
}