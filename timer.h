#ifndef TIMER_H
#define TIMER_H


#define USE_MPIWTIME 1
#define USE_NANOPRECISION 1 // NOTE: If this breaks the build then set it to 0

#if USE_MPIWTIME

#include <mpi.h>

#define TIMER_INIT_COUNTERS(_start_,_stop_)  double _start_, _stop_;


#define TIMER_GET_CLOCK(_counter_)  	{\
  MPI_Barrier(MPI_COMM_WORLD); \
  _counter_= MPI_Wtime();\
  }


#define TIMER_GET_DIFF(_start_,_stop_,_diff_){				\
    _diff_ =((_stop_) - (_start_))*(1000000000); }



#elif USE_NANOPRECISION
/*
  Nanosecond Precision Timer

  http://www.guyrutenberg.com/2007/09/22/profiling-code-using-clock_gettime/
timespec diff(timespec start, timespec end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}
  NOTE: need to link with -lrt

https://stackoverflow.com/questions/3875197/gcc-with-std-c99-complains-about-not-knowing-struct-timespec
*/


#include <time.h>
//TODO: fix diff to match what was in blog
#define TIMER_INIT_COUNTERS(_start_,_stop_)  struct timespec _start_, _stop_;


#define TIMER_GET_CLOCK(_counter_) clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &_counter_);


#define TIMER_GET_DIFF(_start_,_stop_,_diff_){				\
  _diff_ = (_stop_.tv_sec-_start_.tv_sec) * 1000000000 + _stop_.tv_nsec - _start_.tv_nsec;}

  

#else
/*
  Microsecond Precision Timer.
*/

#include <sys/time.h>

#define TIMER_INIT_COUNTERS(_start_,_stop_) struct timeval _start_; struct timeval _stop_;
#define TIMER_GET_CLOCK(_counter_) gettimeofday(&_counter_, NULL);
#define TIMER_GET_DIFF(_start_,_stop_,_diff_) _diff_ = ((_stop_.tv_sec - _start_.tv_sec) * 1e6 + _stop_.tv_usec - _start_.tv_usec)* 1000; // <-- TODO: this 1000 should actually be derived like it is in memory mountatin

#endif // TIMERS

#define TIMER_WARMUP(_start_,_stop_)\
{\
  TIMER_GET_CLOCK(_start_);  TIMER_GET_CLOCK(_stop_);	\
  TIMER_GET_CLOCK(_start_);  TIMER_GET_CLOCK(_stop_);	\
  TIMER_GET_CLOCK(_start_);  TIMER_GET_CLOCK(_stop_);	\
  TIMER_GET_CLOCK(_start_);  TIMER_GET_CLOCK(_stop_);	\
}


#endif /* TIMER_H */
