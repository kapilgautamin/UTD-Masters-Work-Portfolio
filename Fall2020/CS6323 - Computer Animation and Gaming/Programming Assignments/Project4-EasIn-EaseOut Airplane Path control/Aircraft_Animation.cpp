#include "Aircraft_Animation.h"


Aircraft_Animation::Aircraft_Animation()
{
	this->m_model_mat = glm::mat4(1.0f);
}


Aircraft_Animation::~Aircraft_Animation()
{
}

void Aircraft_Animation::init()
{
	reset();
}

void Aircraft_Animation::init(Curve* animation_curve)
{
	m_animation_curve = animation_curve;
	this->curve_length = m_animation_curve->cache[m_animation_curve->cache_size-1].length;
	//std::cout<< "curve length " << this->curve_length << std::endl;
	//for(int i=128;i<145;i++)
	//std::cout << i  << " " << m_animation_curve->cache[i].length << std::endl;
	reset();
}

int Aircraft_Animation::linear_search(float distance)
{
	int pos_cache;
	for (pos_cache = 0; pos_cache < m_animation_curve->cache_size; pos_cache++) {
		if (m_animation_curve->cache[pos_cache].length > distance)
			break;
	}
	return pos_cache;
}

int Aircraft_Animation::binary_search(float distance)
{
	int low = 0;
	int high = m_animation_curve->cache_size;
	int mid;
	while (low < high) {
		mid = low + (high - low) / 2;
		//std::cout << " low " << low << " mid " << mid << " high " << high << std::endl;
		if (low == mid)
			return mid;

		if (m_animation_curve->cache[mid].length < distance)
			low = mid;
		else
			high = mid;
	}
	return mid;
}

float Aircraft_Animation::ease(float curr_time)
{
	float distance = 0.0f;
	if (curr_time <= this->t1) {
		// d = v0 * (t^2 / 2 * t1)
		distance = this->max_speed * (powf(curr_time,2) / (2*this->t1));
	}
	else if (curr_time <= this->t2) {
		distance = this->max_speed * (this->t1/2) + this->max_speed * (curr_time - this->t1);
	}
	else {
		float numerator = curr_time - this->t2;
		float denominator = 2*(1 - this->t2);
		distance = this->max_speed * (this->t1 / 2) + this->max_speed * (this->t2 - this->t1) + this->max_speed * (1.0 - (numerator / denominator)) * (curr_time - this->t2);
	}
	return distance * this->curve_length;
}

void Aircraft_Animation::update(float delta_time)
{
	this->max_speed = 2 / (this->t2 - this->t1 + 1);

	this->moving_time += delta_time;
	if (this->moving_time < this->total_moving_time) {
		float distance = ease(this->moving_time/this->total_moving_time);

		int low_point_pos = binary_search(distance);
		int high_point_pos = low_point_pos + 1;

		//int high_point_pos = linear_search(distance);
		//int low_point_pos = high_point_pos - 1;
		//std::cout << "curr time " << this->moving_time << " curr distance " << distance << " ls " << low_point_pos << " bs " << bs << std::endl;

		glm::vec3 low_point = m_animation_curve->cache[low_point_pos].point;
		glm::vec3 high_point = m_animation_curve->cache[high_point_pos].point;

		glm::vec3 interpolated_point = (1 - this->moving_time) * low_point + this->moving_time * high_point;

		m_model_mat = glm::mat4(1.0f);
		m_model_mat = glm::translate(m_model_mat, interpolated_point);
	}

}

void Aircraft_Animation::reset()
{
	m_model_mat = glm::mat4(1.0f);
	this->moving_time = 0.0f;
	if (m_animation_curve != nullptr && m_animation_curve->control_points_pos.size() > 0)
	{
		m_model_mat = glm::translate(m_model_mat, m_animation_curve->control_points_pos[0]);
	}
}
