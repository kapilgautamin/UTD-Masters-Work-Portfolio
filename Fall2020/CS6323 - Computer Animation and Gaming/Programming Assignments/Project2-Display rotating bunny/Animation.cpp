#include "Animation.h"

Animation::Animation()
{
	this->m_model_mat = glm::mat4(1.0f);
}

Animation::~Animation()
{
}

void Animation::init()
{
	reset();
}

void Animation::update(float delta_time)
{
}

void Animation::reset()
{
	m_model_mat = glm::mat4(1.0f);
	m_model_mat = glm::translate(m_model_mat, glm::vec3(5.0f, 0.0f, 0.0f));
}

void Animation::rotateLCS(float angle)
{
	std::cout << "Rotate local X by " << angle << " degrees." << std::endl;
	glm::vec3 along_x_axis = glm::vec3(1.0f, 0.0f, 0.0f);
	m_model_mat = glm::rotate(m_model_mat, glm::radians(angle), along_x_axis);
}

void Animation::rotateWCS(float angle)
{
	std::cout << "Rotate global Y by " <<  angle << " degrees." << std::endl;
	glm::vec3 along_y_axis = glm::vec3(0.0f, 1.0f, 0.0f);

	glm::mat4 rot_mat(1);
	rot_mat = glm::rotate(rot_mat, glm::radians(angle), along_y_axis);
	m_model_mat = rot_mat * m_model_mat;
}