/*
 * main.cpp
 *
 *  Created on: 07/12/2016
 *      Author: jpeumesmo
 */



#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <alsa/asoundlib.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <time.h>

#define CHK(stmt, msg) if((stmt) < 0) {puts("ERROR: "#msg); exit(1);}
#define NOTEVEL 80

int open_client(snd_seq_t** handle, int* port_id){
	CHK(snd_seq_open(handle, "default", SND_SEQ_OPEN_OUTPUT, 0), "Could not open sequencer");
	CHK(snd_seq_set_client_name(*handle, "Chords Client"), "Could not set client name");
	CHK(*port_id = snd_seq_create_simple_port(*handle, "Out", SND_SEQ_PORT_CAP_READ|SND_SEQ_PORT_CAP_SUBS_READ, SND_SEQ_PORT_TYPE_APPLICATION), "Could not open port");
}

void send_note(unsigned char vel, unsigned char note, int channel, snd_seq_t* handle, int port_id){
//Declaração do evento de saida
	snd_seq_event_t out_ev;
//Este trecho é necessário para preparar um evento para envio
	snd_seq_ev_clear(&out_ev);
	snd_seq_ev_set_source(&out_ev, port_id);
	snd_seq_ev_set_subs(&out_ev);
	snd_seq_ev_set_direct(&out_ev);
	snd_seq_ev_set_fixed(&out_ev);
//Se a velocidade do evento for 0, significa que ele é do tipo NOTEOFF
	if(vel == 0){
		out_ev.type = SND_SEQ_EVENT_NOTEOFF;
	}else{
		out_ev.type = SND_SEQ_EVENT_NOTEON;
	}
	out_ev.data.note.velocity = vel;
	out_ev.data.note.channel = channel;
	out_ev.data.note.note = note;
//Os dois comandos abaixo são utilizados para fazer o envio do evento criado
	snd_seq_event_output(handle, &out_ev);
	snd_seq_drain_output(handle);
}


namespace patch{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}

using namespace cv;

String exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL)
                result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

void locate(String &buff){
  std::string in;
  char aux[256];
  in = exec("locate data/haarcascades/haarcascade_frontalface_alt.xml");
  buff = in;
}


void TSL (Mat& in, Mat& out){

	for (int i = 0; i < in.rows; i++){
		for (int j = 0; j < in.cols; j++){

			cv::Vec3f intensity = in.at<Vec3f>(i,j);
			float blue=intensity.val[0];
			float green=intensity.val[1];
			float red=intensity.val[2];

			//SOMATORIO
			double sum=red+blue+green;

			//NORMALIZADO
			double r= red/sum;
			double g= green/sum;

			//R' E G'
			double r_ = r - (1/3);
			double g_ = g - (1/3);


			//CALCULA O L
			double L = (0.299*red) + (green*0.587) + (blue*0.114);


			//CALCULA O S
			double S = sqrt((9/5)*(pow(r_,2)+pow(g_,2)));

			//CALCULA O T
			double T;
			if (g_ == 0) {
				T = 0;
			}
			if (g_ < 0) {
				T = ( (1/(2*3.14)) * atan((r_/g_) + (3/4) ) ) ;
			}
			if (g_ > 0) {
				T = ( (1/(2*3.14)) * atan((r_/g_) + (1/4) ) ) ;
			}
			/* MONTA A SAIDAx

			A.data[A.step[0]*i + A.step[1]* j + 0] = (b*255);
				 A.data[A.step[0]*i + A.step[1]* j + 1] = (g*255);
				 A.data[A.step[0]*i + A.step[1]* j + 2] = (r*255);
				*/

			out.data[out.step[0]*i + out.step[1]*j + 0] = T;
			out.data[out.step[0]*i + out.step[1]*i + 1] = S;
			out.data[out.step[0]*i + out.step[1]*i + 2] = L;
		}
	}
  //imshow("teste",T);
	imshow("teste",out);
}

void balancear(Mat& in, Mat& out, float percent) {
	assert(in.channels() == 3);
	assert(percent > 0 && percent < 100);

	float half_percent = percent / 200.0f;

	std::vector<Mat> tmpsplit; split(in,tmpsplit);
	for(int i=0;i<3;i++) {
		//find the low and high precentile values (based on the input percentile)
		Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
		cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
		int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
		int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
		// std::cout << lowval << " " << highval << "\n";

		//saturate below the low percentile and above the high percentile
		tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
		tmpsplit[i].setTo(highval,tmpsplit[i] > highval);

		//scale the channel
		normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
	}
	merge(tmpsplit,out);
}

bool compareFacePosition ( const Rect & face1, const Rect & face2 ) {
	int x1 = face1.x;
	int x2 = face2.x;
	//double i = fabs( face1.area() );
	//double j = fabs( face2.area() );
	return ( x1 > x2 );
}

bool compareContourAreas ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	double i = fabs( contourArea(cv::Mat(contour1)) );
	double j = fabs( contourArea(cv::Mat(contour2)) );
	return ( i > j );
}

bool compareContourPosition(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	int x1 = contour1.at(2).x;
	int x2 = contour2.at(2).x;
	return (x1 > x2);
}

bool compareContourConvexity ( std::vector<cv::Point> contour1, std::vector<cv::Point> contour2 ) {
	std::vector<std::vector<int> >hull (2);
	std::vector<std::vector<Vec4i> > defects(2);


	convexHull(contour1, hull[0], false);
	convexHull(contour2, hull[1], false);
	convexityDefects(contour1, hull[0], defects[0]);
	convexityDefects(contour2, hull[1], defects[1]);

	int i = defects[0].size();
	int j = defects[1].size();

	return ( i > j );
}

void binarizar(Mat &in, Mat& out){

	Mat element = getStructuringElement( 0,
			Size( 2*3+ 1, 2*3+1 ),
			Point( 3, 3) );Mat kernel = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);


	Mat blured, hsv, balanceada;



	blur( in, blured, Size(13,13) );
//  blur( in, blured, Size(17,17) );


	//balancear(blured,balanceada,10);

	//cvtColor(balanceada,hsv,CV_BGR2HSV);
	cvtColor(blured, hsv, CV_BGR2HSV);
//	balancear(hsv,balanceada,10);
	//cvtColor(in,hsv,CV_BGR2HSV);

//  inRange(balanceada, Scalar(0, 40, 60), Scalar(15, 150, 255), out);

	inRange(hsv, Scalar(0, 40, 60), Scalar(15, 150, 255), out);
//  inRange(hsv, Scalar(0, 48, 80), Scalar(20, 255, 255), out);

  morphologyEx(out,out,1,element);
  //morphologyEx( out, out, MORPH_CLOSE, element );

	//std::vector<Mat> channels;
	//split(hsv,channels);
	//imshow("H",channels[0]);
	//imshow("S",channels[1]);
	//imshow("V",channels[2]);
}

//void tracking(Ptr<Tracker> trackerFace,Rect2d faceBox,Ptr<Tracker> trackerLeft,Rect2d leftBox,Ptr<Tracker> trackerRigth,Rect2d rigthBox ){

  //bbox = boundingRect( Mat(contours[0]) );
  //tracker->update(frame, bbox);
  //rectangle(frame, bbox, azul, 2, 1 );

//}

int main( int argc, char** argv ){

//  FILE *esquerda,*direita;
//  esquerda = fopen("esquerda.txt","w");
//  direita = fopen("direita.txt","w");
	snd_seq_t* handle;
	//snd_seq_t* handle1;
	// snd_seq_t* handle2;
	// snd_seq_t* handle3;
  int ne=0,nd=0, i, portaid;
	char key;


  int base = 60;
	open_client(&handle, &portaid);
/*
  open_client(&handle, &direitax);
  open_client(&handle1, &direitay);
  open_client(&handle2, &esquerdax);
	open_client(&handle3, &esquerday);
*/
  int contArq = 0;
  time_t rawtime;
struct tm * timeinfo;
int flagTrack = 0;
int exPtIniTrack,radiusIniTrack, eyPtIniTrack,contTrack =0;
int dxPtIniTrack, dyPtIniTrack;
int cont;
int flagInt = 0;
int posXRec, posYRec;
Mat kernel = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, -1, 1,
        0, 1, 0);


  //VARIAVEIS
	Scalar verde = Scalar(0,255,0);
	Scalar azul = Scalar(255,0,0);
	Scalar vermelho = Scalar(0,0,255);
	CascadeClassifier face_cascade;
    Mat frame,gray,bin,aux,hsv,blured;
	int cntr = 0;
    unsigned int numeroPessoas;
    String aux_string, localCascade;


//  Ptr<Tracker> trackerLeft = TrackerTLD::create();
//  Ptr<Tracker> trackerRigth = TrackerTLD::create();

  Ptr<Tracker> trackerLeft = TrackerKCF::create();
  Ptr<Tracker> trackerRigth = TrackerKCF::create();

//  Rect2d leftBox,rigthBox;

//  trackerLeft->init(frame,leftBox);
//  trackerRigth->init(frame,rigthBox);

  //ACHA  PATH DO CASCADE
  locate(aux_string);
  localCascade = aux_string.substr (0, aux_string.length()-1 );

	//INCIALIZA O DISPOSITIVO
	VideoCapture cap(0);


	//VERIFICA SE O DISPOSITIVO FOI INICIALIZADO CORRETAMENTE
	if(!cap.open(0)){
		return 0;
	}

	//LOOP PRINCIPAL
	while(1){
		//capturar frame
    cap >> frame;


    flip(frame, frame, 1);

		cvtColor(frame,gray,CV_BGR2GRAY);
    equalizeHist(gray, gray );


    /*if( !face_cascade.load( "/home/jpeumesmo/Applications/OpenCV/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_alt.xml") ){
      printf("--(!)Error loading\n");
      return -1;
    }*/
    if( !face_cascade.load("haarcascade_frontalface_alt.xml") ){
      printf("--(!)Error loading\n");
      return -1;
    }
		//face_cascade.load( "haarcascade_frontalface_alt.xml" ) ;

		std::vector<Rect> faces;

		face_cascade.detectMultiScale( gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

		numeroPessoas = faces.size();

    aux = frame.clone();
		binarizar(aux,bin);
		blur( bin, blured, Size(7,7) );
//		imshow("Binaria",blured);

		std::vector<std::vector<Point> > contours;
		std::vector<Vec4i> hierarchy;
    std::vector<Moments> mu(2);
    std::vector<Point2f> mc(2);
    std::vector<std::vector<Point> > hull(2);
    std::vector<std::vector<int> > hulli(2);
    std::vector<std::vector<Vec4i> > defects(2);


		findContours( bin, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		std::sort(contours.begin(), contours.end(), compareContourAreas);



//    std::cout << flagInt << '\n';
		switch (numeroPessoas){


		case(0):{
			/*
			 * Caso de nenhuma pessoa detectada
			 */
			putText(frame,"Nenhuma pessoa detectada",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
			break;



		}
		case(1):{
			/*
			 * Caso de uma pessoa detectada
			 */

			sort(contours.begin(),contours.begin()+3,compareContourPosition);

			Point centerUnico( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );
			ellipse( frame, centerUnico, Size( faces[0].width*0.5, faces[0].height*0.5), 0, 0, 360, verde, 4, 8, 0 );
      if (!flagTrack){
        exPtIniTrack = faces[0].x - faces[0].width*0.5;
        eyPtIniTrack = frame.size().height*0.2;
        dxPtIniTrack = faces[0].x + faces[0].width*0.5;
        dyPtIniTrack = frame.size().height*0.2;
        radiusIniTrack = faces[0].width*0.25;
        //Point diniciarTrack(faces[0].x+faces[0].width*0.2,frame.size().height*0.5 );
        //Point einiciarTrack(faces[0].x+faces[0].width*0.2,frame.size().height*0.5 );
        flagTrack = 1;
      }

      /*
       PEGAR CENTRO DE MASSA
       MAO DIREITA COR AZUL INDICE 0
       MAO ESQUERDA COR VERMELHA INDICE 2
      */

      mu[0] = moments( contours[0], false );
      mu[1] = moments( contours[2], false );
      mc[0] = Point2f( mu[0].m10/mu[0].m00 , mu[0].m01/mu[0].m00 );
      mc[1] = Point2f( mu[1].m10/mu[1].m00 , mu[1].m01/mu[1].m00 );

/*
    convexHull( Mat(contours[0]), hull[0], false );
    convexHull( Mat(contours[2]), hull[1], false );

    convexHull( Mat(contours[0]), hulli[0], false );
    convexHull( Mat(contours[2]), hulli[1], false );


    convexityDefects( Mat(contours[0]), hulli[0], defects[0] );
    convexityDefects( Mat(contours[2]), hulli[1], defects[1] );

//      morphologyEx(Mat(contours[0]), Mat(hull[0]), MORPH_HITMISS, kernel);
//      morphologyEx(Mat(contours[2]), Mat(hull[1]), MORPH_HITMISS, kernel);

//    Point diniciarTrack(faces[0].x - faces[0].width*0.5,dyPtIniTrack);
//    circle(frame,diniciarTrack,radiusIniTrack,vermelho,1,8,0);
//    Point einiciarTrack(faces[0].x + faces[0].width*1.5,eyPtIniTrack);
//    circle(frame,einiciarTrack,radiusIniTrack,azul,1,8,0);

//------------------------------------------------------------------------------
  */
    //  drawContours(frame,contours,0,verde,-1,8,hierarchy);
    //  drawContours(frame,contours,2,verde,-1,8,hierarchy);

      circle( frame, mc[0], 8, azul, -1, 8, 0 );
    //  drawContours( frame, hull, 1, azul, 1, 8, std::vector<Vec4i>(), 0, Point() );

// std::cout <<hull[1].size() << "\t"<< defects[1][1].size() << '\n';
//for (int k = 0; k < hull[1].size(); k++){
/*
          cont = 0;
            for (int j = 0; j < defects[1].size(); j++){
/*
  //            if (defects[1][j][3] > 20 * 256 /*filter defects by depth*///){
/*
                int ind_0 = defects[1][j][0];//start point
                int ind_1 = defects[1][j][1];//end point
                int ind_2 = defects[1][j][2];//defect point
                if (contours[2][ind_0].y < mc[1].y){
                    cont++;
                //    circle(frame, contours[2][ind_0], 5, Scalar(0, 0, 255), -1);
            //circle(frame, contours[2][ind_1], 5, Scalar(255, 0, 0), -1);
                //    circle(frame, contours[2][ind_2], 5, Scalar(0, 255, 0), -1);
                //    line(frame, contours[2][ind_2], contours[2][ind_0], Scalar(0, 255, 255), 1);
                //    line(frame, contours[2][ind_2], contours[2][ind_1], Scalar(0, 255, 255), 1);

                }
              }
            }

            if (flagInt == 6){
            //desenha retangulo solto
              rectangle(frame,Point(posXRec-100,posYRec-100),Point(posXRec+100,posYRec+100),azul,-1,8,0);
            }
            if (flagInt == 4){
              //desenha retangulo durante percurso
              rectangle(frame,Point(mc[1].x-100,mc[1].y-100),Point(mc[1].x+100,mc[1].y+100),azul,-1,8,0);
            }

            switch (cont) {
              case 5:{
                putText(frame,"5 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
                if (flagInt != 4 && mc[1].x < 230 && mc[1].x > 30 && mc[1].y < 330 && mc[1].y > 140){
                  rectangle(frame,Point(mc[1].x-100,mc[1].y-100),Point(mc[1].x+100,mc[1].y+100),azul,-1,8,0);
                  flagInt = 2;
                }
                if (flagInt == 4 ){
                  //if para soltar o objeto
                  Point2f aux = mc[1];
                  rectangle(frame,Point(aux.x-100,aux.y-100),Point(aux.x+100,aux.y+100),azul,-1,8,0);
                  posXRec = aux.x;
                  posYRec = aux.y;
                  flagInt = 6;
                }
              }

              break;

              case 4:{
                putText(frame,"4 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 3:{
                putText(frame,"3 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 2:{
                putText(frame,"2 dedos",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 1:{
                putText(frame,"1 dedo",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
              }

              break;

              case 0:{
                putText(frame,"Nenhum dedo",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);
                if (flagInt == 2 && mc[1].x < 230 && mc[1].x > 30 && mc[1].y < 330 && mc[1].y > 140){
                  // if para segurar o objeto
                  rectangle(frame,Point(mc[1].x-100,mc[1].y-100),Point(mc[1].x+100,mc[1].y+100),azul,-1,8,0);
                  flagInt = 3;
                }
                if (flagInt == 3){
                  //if para desenha o retangulo junto ao centro de massa
                  rectangle(frame,Point(mc[1].x-100,mc[1].y-100),Point(mc[1].x+100,mc[1].y+100),azul,-1,8,0);
                  flagInt = 4;
                }
              }

              break;

              default:{
                break;
              }
            }
  //        }
/*
      for (const Vec4i& v : defects[1]){
        int startidx = v[0]; Point ptStart(contours[0][startidx]);
        int endidx = v[1]; Point ptEnd(contours[0][endidx]);
        int faridx = v[2]; Point ptFar(contours[0][faridx]);

        line(frame,ptStart,ptEnd,vermelho,1);
        line(frame,ptStart,ptFar,vermelho,1);
        line(frame,ptEnd,ptFar,vermelho,1);
        circle(frame,ptFar,4,vermelho,2);
      }
*/
      //drawContours(frame,contours,2,verde,-1,8,hierarchy);
      circle( frame, mc[1], 8, vermelho, -1, 8, 0 );

      //if ( ((mc[1].x - dxiniciarTrack.x) * (mc[1].x - diniciarTrack.x)) + ((mc[1].y - diniciarTrack.y) * (mc[1].y - diniciarTrack.y)) < (radiusIniTrack * radiusIniTrack )
      //    &&  ((mc[0].x - diniciarTrack.x) * (mc[0].x - diniciarTrack.x)) + ((mc[0].y - diniciarTrack.y) * (mc[0].y - diniciarTrack.y)) < (radiusIniTrack * radiusIniTrack )

//    ){
//        std::cout << contTrack << '\n';
  //      contTrack++;
  //      if (contTrack == 3){
    //      std::cout << "tracking" << '\n';
    //    }
    //  }
    //  drawContours( frame, hull, 1, azul, 1, 8, std::vector<Vec4i>(), 0, Point() );

//      time ( &rawtime );
//  timeinfo = localtime ( &rawtime );
//  fprintf ( "%d    Data atual do sistema é: %s ",rawtime , asctime (timeinfo));

//      fprintf(esquerda, "%d %lf %lf \n", contArq,mc[1].x,mc[1].y);
//      fprintf(direita, "%d %lf %lf \n", contArq,mc[0].x,mc[0].y);
//      contArq++;
		std::cout<<"size\t"<<frame.size();
    std::cout<<"Esquerda x:\t"<< int(((mc[1].x/480))*127) << "\tEsquerda y:\t" << int(mc[1].y)<<"\n";
    std::cout<<"Direita x:\t"<< int(((mc[0].x/848))*127) << "\tDireita y:\t" <<int(mc[0].y)<<"\n";

		send_note(0, ne, 0, handle, portaid);
    send_note(0, nd, 0, handle, portaid);

//		send_note(80 , int(mc[1].x)/10, 0, handle, portaid);
//    send_note(80, int(mc[0].x)/10, 0, handle, portaid);


    send_note(int(((mc[1].y/480))*127), int(((mc[1].x/480))*127), 0, handle, portaid);
    send_note(int(((mc[0].y/848))*127), int(((mc[0].x/848))*127), 0, handle, portaid);
		ne = int(((mc[1].x/480))*127);
		nd = int(((mc[0].x/848))*127);

      break;
		}

		default:{
			/*
			 * Mais do que 3 pessoas detectadas
			 */
			putText(frame,"Pessoas demais detectadas",Point(200,50),FONT_HERSHEY_SIMPLEX,1,vermelho,1,8);

			break;
		}
		}

		imshow("frame",frame);

		cntr++;
//		imwrite("/home/jpeumesmo/workspace/Rosto/images/bin/"+patch::to_string(cntr)+".jpg",bin);
//    imwrite("/home/jpeumesmo/workspace/IC/cm/imagens/b/f"+patch::to_string(cntr)+".jpg",bin);
  //  imwrite("/home/jpeumesmo/workspace/IC/cm/imagens/f"+patch::to_string(cntr)+".jpg",frame);
    imwrite("/home/jpeumesmo/UFSJ/Congressos/ubimus/MIDI-Project/visao/imagens/f"+patch::to_string(cntr)+".jpg",frame);
		if(waitKey(30) >= 0) {
      break;//QUEBRA LACO PRINCIPAL
//      fclose(direita);
//      fclose(esquerda);
    }
    }

	return 0;
}
