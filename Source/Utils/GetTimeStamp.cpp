#include "GetTimeStamp.h"

//΢�뼶ʱ���
long long getTimeStamp_us() {
    //��1970-01-01 00��00��00�����ڵ�ʱ��
    std::chrono::system_clock::duration duration_since_epoch = std::chrono::system_clock::now().time_since_epoch();
    //ת��Ϊ΢����
    long long microseconds_since_epoch = std::chrono::duration_cast<std::chrono::microseconds>(duration_since_epoch).count();
    return microseconds_since_epoch;
}

//�뼶ʱ���
int getTimeStamp_s() {
    time_t t;
    time(&t);
    return t;
}
