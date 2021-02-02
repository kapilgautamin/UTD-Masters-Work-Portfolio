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
	this->curve_length = m_animation_curve->cache[m_animation_curve->cache_size - 1].length;
	//std::cout<< "curve length " << this->curve_length << std::endl;
	//for(int i=1605;i<m_animation_curve->cache_size;i++)
	//std::cout << i  << " " << m_animation_curve->cache[i].length << std::endl;
	reset();
}

int Aircraft_Animation::binary_search(float distance)
{
	int low = 0;
	int high = m_animation_curve->cache_size - 1;
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
	this->max_speed = 2.0f / (this->t2 - this->t1 + 1);
	float distance = 0.0f;
	if (curr_time <= this->t1) {
		// d = v0 * (t^2 / 2 * this->t1)
		distance = this->max_speed * (pow(curr_time, 2) / (2 * this->t1));
	}
	else if (curr_time <= this->t2) {
		distance = this->max_speed * (this->t1 / 2) + this->max_speed * (curr_time - this->t1);
	}
	else {
		float numerator = curr_time - this->t2;
		float denominator = 2 * (1 - this->t2);
		distance = this->max_speed * (this->t1 / 2) + this->max_speed * (this->t2 - this->t1) + this->max_speed * (1.0 - (numerator / denominator)) * (curr_time - this->t2);
	}
	return distance * this->curve_length;
}

void Aircraft_Animation::update(float delta_time, bool rotation_enabled)
{
	this->moving_time += delta_time;
	if (this->moving_time <= this->total_moving_time) {
		this->curr_dist = ease(this->moving_time / this->total_moving_time);

		int low_point_pos = binary_search(this->curr_dist);
		int high_point_pos = low_point_pos + 1;
		
		int segment_no = m_animation_curve->cache[low_point_pos].segment_no;
		if (this->prev_segment != segment_no) {
			this->prev_segment = segment_no;
			this->prev_segment_start = m_animation_curve->cache[low_point_pos].length;
		}

		float ratio_dist = (this->curr_dist - this->prev_segment_start) / m_animation_curve->segment_length[this->prev_segment];
		//std::cout << "curr time " << this->moving_time << " ratio dist " << ratio_dist << " curr distance " << this->curr_dist << " ls " << low_point_pos << " segment " << m_animation_curve->cache[low_point_pos].segment_no << std::endl;

		glm::vec3 low_point = m_animation_curve->cache[low_point_pos].point;
		glm::vec3 high_point = m_animation_curve->cache[high_point_pos].point;
		glm::vec3 interpolated_point = (1 - ratio_dist) * low_point + ratio_dist * high_point;
		m_model_mat = glm::mat4(1.0f);
		m_model_mat = glm::translate(m_model_mat, interpolated_point);

		if(rotation_enabled) {
			
			glm::quat low_segment_quat = m_animation_curve->control_points_quaternion[segment_no];
			//std::cout << "quat " << to_string(low_segment_quat) << std::endl;
			low_segment_quat = glm::normalize(low_segment_quat);
			//std::cout << "norm quat " << to_string(low_segment_quat) << std::endl;

			glm::quat high_segment_quat = m_animation_curve->control_points_quaternion[(segment_no + 1) % 8];
			high_segment_quat = glm::normalize(high_segment_quat);

			float costheta = glm::dot(low_segment_quat, high_segment_quat);
			
			if (costheta < 0) {
				//std::cout << "negative" << std::endl;
				high_segment_quat *= -1;
				costheta *= -1;
			}

			float angle = acos(costheta);
			glm::quat interpolated_quat = (sin((1 - ratio_dist) * angle) * low_segment_quat + sin(ratio_dist * angle) * high_segment_quat) / sin(angle);

			glm::mat4 rotation_mat = m_animation_curve->quat_to_rotation_mat(interpolated_quat);
			m_model_mat *= rotation_mat;
		}
	}
}

void Aircraft_Animation::reset()
{
	m_model_mat = glm::mat4(1.0f);
	this->moving_time = 0.0f;
	this->curr_dist = 0.0f;
	this->prev_segment = -1;
	if (m_animation_curve != nullptr && m_animation_curve->control_points_pos.size() > 0)
	{
		m_model_mat = glm::translate(m_model_mat, m_animation_curve->control_points_pos[0]);
		glm::mat4 rotation_mat = m_animation_curve->quat_to_rotation_mat(m_animation_curve->control_points_quaternion[0]);
		m_model_mat *= rotation_mat;
	}
}
