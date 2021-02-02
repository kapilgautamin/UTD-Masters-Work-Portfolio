#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>	
#include <Eigen/Dense>
#include <math.h>
#include <glm\gtx\string_cast.hpp>


class Bone_Animation
{
public:
	Bone_Animation();
	~Bone_Animation();

	void init();
	void update(float delta_time, glm::mat4 root_matrix);
	glm::mat4 get_first_bone_matrix(glm::mat4 root_matrix);
	glm::mat4 get_second_bone_matrix(glm::mat4 first_bone_obj_mat);
	glm::mat4 get_third_bone_matrix(glm::mat4 second_bone_obj_mat);
	void reset();

public:

	// Here the head of each vector is the root bone
	std::vector<glm::vec3> scale_vector;
	std::vector<glm::vec3> rotation_degree_vector;
	std::vector<glm::vec4> colors;

	Eigen::MatrixXf jacobian_mat = Eigen::MatrixXf::Zero(3, 9);
	Eigen::MatrixXf jacobian_mat_transpose = Eigen::MatrixXf::Zero(3, 9);
	Eigen::MatrixXf delta_e = Eigen::MatrixXf::Zero(3, 1);
	Eigen::MatrixXf delta_phi = Eigen::MatrixXf::Zero(3, 9);

	glm::vec3 root_position;
	glm::vec3 target_position;
	glm::vec3 current_end_position;
	float threshold = 1e-6;
	bool move_bone;
};

