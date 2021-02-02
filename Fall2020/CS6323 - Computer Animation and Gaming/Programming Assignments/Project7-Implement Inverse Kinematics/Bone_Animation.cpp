#include "Bone_Animation.h"

Bone_Animation::Bone_Animation()
{
}


Bone_Animation::~Bone_Animation()
{
}

void Bone_Animation::init()
{
	root_position = { 2.0f,0.5f,2.0f};
	target_position = { 3.0f,8.0f,3.0f};

	move_bone = false;
	scale_vector =
	{
		{1.0f,1.0f,1.0f},
		{0.5f,4.0f,0.5f},
		{0.5f,3.0f,0.5f},
		{0.5f,2.0f,0.5f}
	};

	rotation_degree_vector =
	{
		{0.0f,0.0f,0.0f},
		{0.0f,0.0f,30.0f},
		{0.0f,0.0f,30.0f},
		{0.0f,0.0f,30.0f}
	};

	colors =
	{
		{0.7f,0.0f,0.0f,1.0f},
		{0.7f,0.7f,0.0f,1.0f},
		{0.7f,0.0f,0.7f,1.0f},
		{0.0f,0.7f,0.7f,1.0f},
		{0.0f,0.85f,0.0f,1.0f}
	};

}

void Bone_Animation::update(float delta_time, glm::mat4 root_matrix)
{
	if (move_bone) {
		/*
		* a’i: unit length rotation axis in world space
		* r’i: position of joint pivot in world space
		* e:  end effector position in world space
		*
		*	  del(e)
		*	---------- =	cross(a'i,(e-r'i))
		*	del(phi,i)
		*
		*
		*/

		//get first, second, third bone matrix
		glm::mat4 first_bone_obj_mat = get_first_bone_matrix(root_matrix);
		glm::mat4 second_bone_obj_mat = get_second_bone_matrix(first_bone_obj_mat);
		glm::mat4 third_bone_obj_mat = get_third_bone_matrix(second_bone_obj_mat);

		glm::vec3 translate_matrix = glm::vec3(0, 0.5f, 0);
		first_bone_obj_mat = glm::translate(first_bone_obj_mat, translate_matrix);
		second_bone_obj_mat = glm::translate(second_bone_obj_mat, translate_matrix);
		third_bone_obj_mat = glm::translate(third_bone_obj_mat, translate_matrix);

		glm::vec3 end_position_third_bone = third_bone_obj_mat[3];
		glm::vec3 end_position_second_bone = second_bone_obj_mat[3];
		glm::vec3 end_position_first_bone = first_bone_obj_mat[3];
		glm::vec3 end_position_root_bone = root_matrix[3];

		glm::vec3 first_bone_diff_end_position = end_position_third_bone - end_position_root_bone;
		glm::vec3 second_bone_diff_end_position = end_position_third_bone - end_position_first_bone;
		glm::vec3 third_bone_diff_end_position = end_position_third_bone - end_position_second_bone;

		glm::vec3 dof[9];
		dof[0] = glm::cross(glm::vec3(first_bone_obj_mat[0]), first_bone_diff_end_position);
		dof[1] = glm::cross(glm::vec3(first_bone_obj_mat[1]), first_bone_diff_end_position);
		dof[2] = glm::cross(glm::vec3(first_bone_obj_mat[2]), first_bone_diff_end_position);

		dof[3] = glm::cross(glm::vec3(second_bone_obj_mat[0]), second_bone_diff_end_position);
		dof[4] = glm::cross(glm::vec3(second_bone_obj_mat[1]), second_bone_diff_end_position);
		dof[5] = glm::cross(glm::vec3(second_bone_obj_mat[2]), second_bone_diff_end_position);

		dof[6] = glm::cross(glm::vec3(third_bone_obj_mat[0]), third_bone_diff_end_position);
		dof[7] = glm::cross(glm::vec3(third_bone_obj_mat[1]), third_bone_diff_end_position);
		dof[8] = glm::cross(glm::vec3(third_bone_obj_mat[2]), third_bone_diff_end_position);

		//Putting data in 3x9 matrix
		jacobian_mat <<
			dof[0].x, dof[1].x, dof[2].x, dof[3].x, dof[4].x, dof[5].x, dof[6].x, dof[7].x, dof[8].x,
			dof[0].y, dof[1].y, dof[2].y, dof[3].y, dof[4].y, dof[5].y, dof[6].y, dof[7].y, dof[8].y,
			dof[0].z, dof[1].z, dof[2].z, dof[3].z, dof[4].z, dof[5].z, dof[6].z, dof[7].z, dof[8].z;

		//now data transposed to 9x3 matrix
		jacobian_mat_transpose = jacobian_mat.transpose();

		glm::vec3 goal = target_position - end_position_third_bone;
		delta_e << goal[0], goal[1], goal[2];

		if (glm::length(goal) <= threshold)
			move_bone = false;

		float alpha[9];
		for (int i = 0; i < 9; i++) {
			glm::vec3 numerator = jacobian_mat_transpose(i) * goal;
			glm::vec3 denominator = jacobian_mat(i) * numerator;

			float num = glm::pow(glm::length(numerator), 2);
			float deno = glm::pow(glm::length(denominator), 2);

			if (deno > 0 && deno >= num)
				alpha[i] = num / deno;
			else
				alpha[i] = 0.1f;
			//std::cout << num << " " << deno << " " << alpha[i] << std::endl;
		}

		delta_phi = jacobian_mat_transpose * delta_e;

		//first bone
		rotation_degree_vector[1][0] += delta_phi(0) * alpha[0];
		rotation_degree_vector[1][1] += delta_phi(1) * alpha[1];
		rotation_degree_vector[1][2] += delta_phi(2) * alpha[2];

		//second bone
		rotation_degree_vector[2][0] += delta_phi(3) * alpha[3];
		rotation_degree_vector[2][1] += delta_phi(4) * alpha[4];
		rotation_degree_vector[2][2] += delta_phi(5) * alpha[5];

		//third bone
		rotation_degree_vector[3][0] += delta_phi(6) * alpha[6];
		rotation_degree_vector[3][1] += delta_phi(7) * alpha[7];
		rotation_degree_vector[3][2] += delta_phi(8) * alpha[8];
	}
}

glm::mat4 Bone_Animation::get_first_bone_matrix(glm::mat4 root_matrix) {
	// Translate * Rotate * Scale * Matrix
	glm::mat4 first_bone_obj_mat = glm::mat4(1.0f);
	first_bone_obj_mat = glm::scale(first_bone_obj_mat, scale_vector[1]);

	glm::mat4 first_bone_rot_mat = glm::mat4(1.0f);
	// Rotation axis required in order - y,z,x 
	// So multiplication should be like Rotx * Rotz * Roty * Matrix
	first_bone_rot_mat = glm::rotate(first_bone_rot_mat, glm::radians(rotation_degree_vector[1][0]), glm::vec3(1, 0, 0));
	first_bone_rot_mat = glm::rotate(first_bone_rot_mat, glm::radians(rotation_degree_vector[1][2]), glm::vec3(0, 0, 1));
	first_bone_rot_mat = glm::rotate(first_bone_rot_mat, glm::radians(rotation_degree_vector[1][1]), glm::vec3(0, 1, 0));
	first_bone_obj_mat = first_bone_rot_mat * first_bone_obj_mat;
	//Whole object is of size 1, so bring the object to origin first
	glm::vec3 first_bone_translate = glm::vec3(0, 0.5f, 0);
	first_bone_obj_mat = glm::translate(first_bone_obj_mat, first_bone_translate);

	root_matrix = glm::translate(root_matrix, glm::vec3(0, 0.5f, 0));
	first_bone_obj_mat = root_matrix * first_bone_obj_mat;
	return first_bone_obj_mat;
}

glm::mat4 Bone_Animation::get_second_bone_matrix(glm::mat4 first_bone_obj_mat) {
	glm::mat4 second_bone_obj_mat = glm::mat4(1.0f);
	second_bone_obj_mat = glm::scale(second_bone_obj_mat, scale_vector[2]);

	glm::mat4 second_bone_rot_mat = glm::mat4(1.0f);
	// Rotation axis required in order - y,z,x 
	// So multiplication should be like Rotx * Rotz * Roty * Matrix
	second_bone_rot_mat = glm::rotate(second_bone_rot_mat, glm::radians(rotation_degree_vector[2][0]), glm::vec3(1, 0, 0));
	second_bone_rot_mat = glm::rotate(second_bone_rot_mat, glm::radians(rotation_degree_vector[2][2]), glm::vec3(0, 0, 1));
	second_bone_rot_mat = glm::rotate(second_bone_rot_mat, glm::radians(rotation_degree_vector[2][1]), glm::vec3(0, 1, 0));
	second_bone_obj_mat = second_bone_rot_mat * second_bone_obj_mat;
	glm::vec3 second_bone_translate = glm::vec3(0, 0.5, 0);
	second_bone_obj_mat = glm::translate(second_bone_obj_mat, second_bone_translate);

	first_bone_obj_mat = glm::translate(first_bone_obj_mat, glm::vec3(0, 0.5, 0));
	first_bone_obj_mat = glm::scale(first_bone_obj_mat, glm::vec3(1.0 / 0.5, 1.0 / 4, 1.0 / 0.5));
	second_bone_obj_mat = first_bone_obj_mat * second_bone_obj_mat;
	return second_bone_obj_mat;
}

glm::mat4 Bone_Animation::get_third_bone_matrix(glm::mat4 second_bone_obj_mat) {
	glm::mat4 third_bone_obj_mat = glm::mat4(1.0f);
	third_bone_obj_mat = glm::scale(third_bone_obj_mat, scale_vector[3]);

	glm::mat4 third_bone_rot_mat = glm::mat4(1.0f);
	// Rotation axis required in order - y,z,x 
	// So multiplication should be like Rotx * Rotz * Roty * Matrix
	third_bone_rot_mat = glm::rotate(third_bone_rot_mat, glm::radians(rotation_degree_vector[3][1]), glm::vec3(0, 1, 0));
	third_bone_rot_mat = glm::rotate(third_bone_rot_mat, glm::radians(rotation_degree_vector[3][2]), glm::vec3(0, 0, 1));
	third_bone_rot_mat = glm::rotate(third_bone_rot_mat, glm::radians(rotation_degree_vector[3][0]), glm::vec3(1, 0, 0));
	third_bone_obj_mat = third_bone_rot_mat * third_bone_obj_mat;

	glm::vec3 third_bone_translate = glm::vec3(0, 0.5, 0);
	third_bone_obj_mat = glm::translate(third_bone_obj_mat, third_bone_translate);

	second_bone_obj_mat = glm::translate(second_bone_obj_mat, glm::vec3(0, 0.5, 0));
	second_bone_obj_mat = glm::scale(second_bone_obj_mat, glm::vec3(1.0 / 0.5, 1.0 / 3, 1.0 / 0.5));

	third_bone_obj_mat = second_bone_obj_mat * third_bone_obj_mat;
	return third_bone_obj_mat;
}

void Bone_Animation::reset()
{
}