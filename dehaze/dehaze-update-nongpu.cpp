
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <omp.h>
#include <deque>
#include <time.h>

#define THREADS 4
using namespace std;
using namespace cv;

inline double maxnum(double a, double b)
{
	if (a >= b)
	{
		return a;
	}
	else
	{
		return b;
	}
}
Mat boxfilter(Mat imSrc, int r)
{
	int hei = imSrc.rows;
	int wid = imSrc.cols;
	Mat imDst = Mat::zeros(imSrc.size(), imSrc.type());
	//cumulative sum over Y axis
	Mat imCum1 = Mat::zeros(imSrc.size(), imSrc.type());
	for (int j = 0; j<imSrc.cols; j++)
	{
		imCum1.at<double>(0, j) = imSrc.at<double>(0, j);
	}
	for (int i = 1; i<imSrc.rows; i++)
	{
		for (int j = 0; j<imSrc.cols; j++)
		{
			imCum1.at<double>(i, j) = imSrc.at<double>(i, j) + imCum1.at<double>(i - 1, j);
		}
	}
	for (int i = 0; i<r + 1; i++)
	{
		for (int j = 0; j<imSrc.cols; j++)
		{
			imDst.at<double>(i, j) = imCum1.at<double>(i + r, j);
		}
	}

	for (int i = r + 1; i<hei - r; i++)
	{
		for (int j = 0; j<imSrc.cols; j++)
		{
			imDst.at<double>(i, j) = imCum1.at<double>(i + r, j) - imCum1.at<double>(i - r - 1, j);
		}
	}

	for (int i = hei - r; i<hei; i++)
	{
		for (int j = 0; j<imSrc.cols; j++)
		{
			imDst.at<double>(i, j) = imCum1.at<double>(hei - 1, j) - imCum1.at<double>(i - r - 1, j);
		}
	}

	//cumulative sum over X axis
	Mat imCum2 = Mat::zeros(imSrc.size(), imSrc.type());
	for (int i = 0; i<imSrc.rows; i++)
	{
		imCum2.at<double>(i, 0) = imDst.at<double>(i, 0);
	}
	for (int i = 0; i<imSrc.rows; i++)
	{
		for (int j = 1; j<imSrc.cols; j++)
		{
			imCum2.at<double>(i, j) = imDst.at<double>(i, j) + imCum2.at<double>(i, j - 1);
		}
	}

	for (int i = 0; i<imSrc.rows; i++)
	{
		for (int j = 0; j<r + 1; j++)
		{
			imDst.at<double>(i, j) = imCum2.at<double>(i, j + r);
		}
	}

	for (int i = 0; i<imSrc.rows; i++)
	{
		for (int j = r + 1; j<wid - r; j++)
		{
			imDst.at<double>(i, j) = imCum2.at<double>(i, j + r) - imCum2.at<double>(i, j - r - 1);
		}
	}

	for (int i = 0; i<imSrc.rows; i++)
	{
		for (int j = wid - r; j<wid; j++)
		{
			imDst.at<double>(i, j) = imCum2.at<double>(i, wid - 1) - imCum2.at<double>(i, j - r - 1);
		}
	}
	return imDst;
}
Mat guidedfilter(Mat I, Mat p, int r, double eps)
{
	int hei = I.rows;
	int wid = I.cols;

	Mat N = boxfilter(Mat::ones(I.size(), I.type()), r);

	Mat mean_I;
	Mat aa = boxfilter(I, r);
	divide(boxfilter(I, r), N, mean_I);

	Mat mean_p;
	divide(boxfilter(p, r), N, mean_p);

	Mat mean_Ip;
	Mat Ip;
	multiply(I, p, Ip);
	divide(boxfilter(Ip, r), N, mean_Ip);

	//this is the covariance of (I, p) in each local patch.
	Mat cov_Ip;
	Mat mean_I_p;
	multiply(mean_I, mean_p, mean_I_p);
	subtract(mean_Ip, mean_I_p, cov_Ip);

	Mat mean_II;
	Mat II;
	multiply(I, I, II);
	divide(boxfilter(II, r), N, mean_II);

	Mat mean_I_I;
	multiply(mean_I, mean_I, mean_I_I);
	Mat var_I;
	subtract(mean_II, mean_I_I, var_I);

	// Eqn. (5) in the paper;
	Mat a;

	var_I.convertTo(var_I, var_I.type(), 1.0, eps);
	divide(cov_Ip, var_I, a);

	// Eqn. (6) in the paper;
	Mat b;
	Mat a_mean_I;
	multiply(a, mean_I, a_mean_I);
	subtract(mean_p, a_mean_I, b);

	double minval, maxval;
	minMaxLoc(b, &minval, &maxval);

	Mat mean_a;
	divide(boxfilter(a, r), N, mean_a);
	Mat mean_b;
	divide(boxfilter(b, r), N, mean_b);

	//Eqn. (8) in the paper;
	Mat q;
	Mat mean_a_I;
	multiply(mean_a, I, mean_a_I);
	add(mean_a_I, mean_b, q);
	return q;

}
void MakeMapping(int* Histgram, float CutLimit = 0.01)
{
	int I, Sum = 0, Amount = 0;
	const int Level = 256;
	for (I = 0; I < Level; I++) Amount += Histgram[I];
	int MinB, MaxB;
	for (I = 0; I < Level; I++)
	{
		Sum = Sum + Histgram[I];
		if (Sum >= Amount * CutLimit)
		{
			MinB = I;
			break;
		}
	}
	Sum = 0;
	for (I = Level - 1; I >= 0; I--)
	{
		Sum = Sum + Histgram[I];
		if (Sum >= Amount * CutLimit)
		{
			MaxB = I;
			break;
		}
	}

	if (MaxB != MinB)
	{
		for (I = 0; I < Level; I++)
		{
			if (I<MinB)
				Histgram[I] = 0;
			else if (I>MaxB)
				Histgram[I] = 255;
			else
				Histgram[I] = 255 * (I - MinB) / (MaxB - MinB);
		}
	}
	else
	{
		for (I = 0; I < Level; I++)
			Histgram[I] = MaxB;        //     this  is must
	}
}

int block = 7;//5
double w1 = 200;//8
double w;
IplImage *src = NULL;
IplImage *dst = NULL;

IplImage* dehaze(IplImage *src, int block, double w)
{
	//rgb channels
	IplImage* dst1 = NULL;
	IplImage* dst2 = NULL;
	IplImage* dst3 = NULL;
	//dst ROI
	IplImage* imgroi1;
	IplImage* imgroi2;
	IplImage* imgroi3;

	IplImage* roidark;

	IplImage* dark_channel = NULL;

	IplImage* toushelv = NULL;
	//dehaze channels
	IplImage* j1 = NULL;
	IplImage* j2 = NULL;
	IplImage* j3 = NULL;
	//combine channels
	IplImage* dst = NULL;
	//image ROI
	CvRect ROI_rect;

	dst1 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	dst2 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	dst3 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

	imgroi1 = cvCreateImage(cvSize(block, block), IPL_DEPTH_8U, 1);
	imgroi2 = cvCreateImage(cvSize(block, block), IPL_DEPTH_8U, 1);
	imgroi3 = cvCreateImage(cvSize(block, block), IPL_DEPTH_8U, 1);
	roidark = cvCreateImage(cvSize(block, block), IPL_DEPTH_8U, 1);

	j1 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	j2 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	j3 = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);
	dark_channel = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

	toushelv = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

	dst = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);

	cvSplit(src, dst1, dst2, dst3, NULL);

	ROI_rect.width = block;
	ROI_rect.height = block;
	ROI_rect.x = 0;
	ROI_rect.y = 0;

	int i, j;
	double min1 = 0;
	double max1 = 0;
	double min2 = 0;
	double max2 = 0;
	double min3 = 0;
	double max3 = 0;
	double min = 0;
	CvScalar value;
	for (i = 0; i<src->width / block; i++)
	{
		for (j = 0; j<src->height / block; j++)
		{
			cvSetImageROI(dst1, ROI_rect);
			cvCopy(dst1, imgroi1, NULL);
			cvMinMaxLoc(imgroi1, &min1, &max1, NULL, NULL);

			cvSetImageROI(dst2, ROI_rect);
			cvCopy(dst2, imgroi2, NULL);
			cvMinMaxLoc(imgroi2, &min2, &max2, NULL, NULL);

			cvSetImageROI(dst3, ROI_rect);
			cvCopy(dst3, imgroi3, NULL);
			cvMinMaxLoc(imgroi3, &min3, &max3, NULL, NULL);

			if (min1<min2)
			{
				min = min1;
			}
			else
			{
				min = min2;
			}
			if (min>min3)
			{
				min = min3;
			}
			//min is ROI Jdrak
			value = cvScalar(min, min, min, min);

			cvSetImageROI(dark_channel, ROI_rect);

			cvSet(roidark, value);

			cvCopy(roidark, dark_channel, NULL);

			cvResetImageROI(dst1);
			cvResetImageROI(dst2);
			cvResetImageROI(dst3);
			cvResetImageROI(dark_channel);

			ROI_rect.x = block*i;
			ROI_rect.y = block*j;


		}
	}

	cvSaveImage("dark_channel.bmp", dark_channel);

	double min_dark;
	double max_dark;

	CvPoint min_loc;
	CvPoint max_loc;
	cvMinMaxLoc(dark_channel, &min_dark, &max_dark, &min_loc, &max_loc, NULL);
	ROI_rect.x = max_loc.x;
	ROI_rect.y = max_loc.y;

	double A_dst1;
	double dst1_min;

	double A_dst2;
	double dst2_min;

	double A_dst3;
	double dst3_min;

	cvSetImageROI(dst1, ROI_rect);
	cvCopy(dst1, imgroi1, NULL);
	cvMinMaxLoc(imgroi1, &dst1_min, &A_dst1, NULL, NULL);

	cvSetImageROI(dst2, ROI_rect);
	cvCopy(dst2, imgroi2, NULL);
	cvMinMaxLoc(imgroi2, &dst2_min, &A_dst2, NULL, NULL);

	cvSetImageROI(dst3, ROI_rect);
	cvCopy(dst3, imgroi3, NULL);
	cvMinMaxLoc(imgroi3, &dst3_min, &A_dst3, NULL, NULL);

	int k, l;
	CvScalar m, n;
	for (k = 0; k<src->height; k++)
	{
		for (l = 0; l<src->width; l++)
		{
			m = cvGet2D(dark_channel, k, l);
			n = cvScalar(255 - w*m.val[0]);
			cvSet2D(toushelv, k, l, n);
		}
	}

	cvSaveImage("tousgelv.jpg", toushelv);

	int p, q;
	double tx;
	double jj1, jj2, jj3;
	CvScalar ix, jx;
	for (p = 0; p<src->height; p++)
	{
		for (q = 0; q<src->width; q++)
		{
			tx = cvGetReal2D(toushelv, p, q);
			tx = tx / 255;
			if (tx<0.1)
			{
				tx = 0.1;
			}
			ix = cvGet2D(src, p, q);
			jj1 = (ix.val[0] - A_dst1) / tx + A_dst1;
			jj2 = (ix.val[1] - A_dst2) / tx + A_dst2;
			jj3 = (ix.val[2] - A_dst3) / tx + A_dst3;
			jx = cvScalar(jj1, jj2, jj3, 0.0);
			cvSet2D(dst, p, q, jx);
		}
	}
	cvSaveImage("dehaze.bmp", dst);

	cvReleaseImage(&dst1);
	cvReleaseImage(&dst2);
	cvReleaseImage(&dst3);
	cvReleaseImage(&imgroi1);
	cvReleaseImage(&imgroi2);
	cvReleaseImage(&imgroi3);
	cvReleaseImage(&roidark);
	cvReleaseImage(&dark_channel);
	cvReleaseImage(&toushelv);
	cvReleaseImage(&j1);
	cvReleaseImage(&j2);
	cvReleaseImage(&j3);


	return dst;
}

Mat dehaze_cplusplus(Mat& img, double kenlRatio, int minAtomsLight)
{
	if(img.empty())
	{
	cout<< "empty frame " << endl;
	return img;
	}
	
	Mat imgcopy = img.clone();
	int w = img.cols;//width
	int h = img.rows;//height

	/****************************************/
	Mat dc;
	dc = Mat::zeros(h, w, CV_8UC1);//dc stores min of channels
	//uchar minval;
	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{

			dc.at<uchar>(i, j) = (uchar)min((int)img.at<Vec3b>(i, j)[0], min((int)img.at<Vec3b>(i, j)[1], (int)img.at<Vec3b>(i, j)[2]));
		}
	}
	/****************************************/

	int krnlsz;

	krnlsz = (int)floor(max(15.0, max(w*kenlRatio, h*kenlRatio)));
	/***************************************/
	double tmpmin;
	Mat tmp_img;
	Mat dc2;
	dc2.create(h, w, CV_8UC1);
	#pragma omp parallel for collapse(2) 
	for (int i = 0; i<h; i++)
	{	
	 
		for (int j = 0; j<w; j++)
		{
			getRectSubPix(dc, Size(krnlsz, krnlsz), Point(j, i), tmp_img);
			minMaxLoc(tmp_img, &tmpmin);
			dc2.at<uchar>(i, j) = (uchar)tmpmin;
		}
	}
	/***************************************/
	Mat t;
	t.create(h, w, CV_8UC1);
	Mat M = Mat::ones(h, w, CV_8UC1);
	Mat M2 = M.mul(M, 255);
	subtract(M2, dc2, t);
	double maxval;
	minMaxLoc(dc2, NULL, &maxval);
	double A = min(minAtomsLight, (int)maxval);
	t.convertTo(t, CV_64FC1, 1.f / 255);//convert t to[0,1]
	Mat t_d = t.clone();//[0,1]


	Mat J = Mat::zeros(h, w, CV_64FC3);

	img.convertTo(img, CV_64FC3);
	Mat img_d = img.clone();
	Mat tmp1;
	tmp1 = Mat::ones(h, w, CV_64FC1);
	subtract(tmp1, t_d, tmp1);
	int r = krnlsz * 4;//r is no less than  4x radix of minfilter
	A += r;
	Mat tmp3 = tmp1.mul(tmp1, A);
	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
			if (t_d.at<double>(i, j)<0.1)
			{
				t_d.at<double>(i, j) = 0.1;
			}
			J.at<Vec3d>(i, j)[0] = (double)((double)img_d.at<Vec3d>(i, j)[0] - tmp3.at<double>(i, j)) / t_d.at<double>(i, j);
			J.at<Vec3d>(i, j)[1] = (double)((double)img_d.at<Vec3d>(i, j)[1] - tmp3.at<double>(i, j)) / t_d.at<double>(i, j);
			J.at<Vec3d>(i, j)[2] = (double)((double)img_d.at<Vec3d>(i, j)[2] - tmp3.at<double>(i, j)) / t_d.at<double>(i, j);
		}
	}
	
	double eps = 0.000001f;
	Mat gray;
	cvtColor(imgcopy, gray, CV_RGB2GRAY);
	gray.convertTo(gray, CV_64FC1, 1.f / 255);
	Mat filtered = guidedfilter(gray, t_d, r, eps);
	t_d = filtered.clone();
	double minval1, maxval1;
	minMaxLoc(t_d, &minval1, &maxval1);
	Mat tmp2;
	tmp2 = Mat::ones(h, w, CV_64FC1);
	subtract(tmp2, t_d, tmp2);
	double temp;

	Mat tmp4 = tmp2.mul(tmp2, A);
	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
			if (t_d.at<double>(i, j)<0.1)
			{
				t_d.at<double>(i, j) = 0.1;
			}
			J.at<Vec3d>(i, j)[0] = (double)((double)img_d.at<Vec3d>(i, j)[0] - tmp4.at<double>(i, j)) / t_d.at<double>(i, j);
			J.at<Vec3d>(i, j)[1] = (double)((double)img_d.at<Vec3d>(i, j)[1] - tmp4.at<double>(i, j)) / t_d.at<double>(i, j);
			J.at<Vec3d>(i, j)[2] = (double)((double)img_d.at<Vec3d>(i, j)[2] - tmp4.at<double>(i, j)) / t_d.at<double>(i, j);
		}
	}
	J.convertTo(J, CV_8UC3);

	return J;
}
int main(int argc, char * argv[])
{
	////////////////////////////// Video capture
	
	VideoCapture vid(argv[1]);

	if (!vid.isOpened())
	{
		std::cout << "Error";
		return 0;
	}
	////////////////////////
  	cout << "frame count is," << vid.get(CV_CAP_PROP_FRAME_COUNT) << endl;	

 	clock_t t1,t2;
	t1=clock();
	double kenlRatio = 0.01;
	int minAtomsLight = 220;//max

	srand(time(NULL) + 1);
	int val = rand() % 1000000;
	ostringstream buff;
	buff << val;
	const string out = "vidout" + buff.str() + ".avi";

	VideoWriter output; 
	int ex = static_cast<int>(vid.get(CV_CAP_PROP_FOURCC));
	int h= vid.get(CAP_PROP_FRAME_HEIGHT);
	int w = vid.get(CAP_PROP_FRAME_WIDTH);
	int fps = vid.get(CAP_PROP_FPS);	
	Mat images[THREADS];
	Mat write[THREADS];
	output.open(out, ex, fps, Size(w, h), true);
	Mat img;
	double total = 0;
	
for (;;)
{

	for(int i=0;i<THREADS;i++) 
	{
		vid.read(img);
		if(img.empty())
		break;
		images[i]=img.clone();
	}
	clock_t clocks[THREADS*2];
	#pragma omp parallel for
	for(int i=0;i<THREADS;i++)
	{
		clocks[i] = clock();	
		images[i]=dehaze_cplusplus(images[i],kenlRatio, minAtomsLight);
		clocks[i+1] = clock();
		float diff = float(clocks[i+1] - clocks[i])/CLOCKS_PER_SEC;
		cout << "Time for one image in seconds," << diff << endl;
		total += diff;
	}

	for(int i=0;i<THREADS;i++)
	{
	if(!(images[i].empty()))
	{
		output.write(images[i]);
	}
	} 
		if (img.empty())
			break;
}
	output.release();
	t2=clock();
  	cout << "frame per second is," << vid.get(CV_CAP_PROP_FRAME_COUNT)/total << endl;	

	return 0;
}
