package com.example.myapplication;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private Button bt1;
    private ImageButton imgbt1;
    private Mat mat1;
    private Mat mat2;
    private Mat mat3;
    private Bitmap bitmap1;
    @Override
    protected void onDestroy() {
        super.onDestroy();
        mat1.release();
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i("TAG", "123");
        setContentView(R.layout.activity_main);
        initLoadOpenCv();
        bt1=findViewById(R.id.button);
        bt1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Get drawable resource
                ImageView resultImageView = findViewById(R.id.imageButton);
                Context context = getApplicationContext();
                Resources resources = context.getResources();
                @SuppressLint("ResourceType") InputStream inputStream = resources.openRawResource(R.drawable.a);
                File outputFile = new File(getFilesDir(), "a.png");
                try {
                    OutputStream outputStream = new FileOutputStream(outputFile);
                    byte[] buffer = new byte[1024];
                    int length;
                    while ((length = inputStream.read(buffer)) > 0) {
                        outputStream.write(buffer, 0, length);
                    }
                    outputStream.close();
                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
               Mat img = Imgcodecs.imread(outputFile.getAbsolutePath());
               img.convertTo(img, CvType.CV_32FC3, 1.0 / 255.0);
                long startTime = System.currentTimeMillis();
                Mat res = new Mat(img.size(), CvType.CV_32FC3);
                // 设置通道的大小
                List<Mat> channels = new ArrayList<>();
                Size channelSize =  img.size();

                // 创建具有指定大小和类型的新通道，并添加到 channels 列表中
                for (int k = 0; k < 3; k++) {
                    Mat channel = Mat.zeros(channelSize, CvType.CV_32FC1);
                    channels.add(channel);
                }
                for (int k = 0; k < 3; k++) {
                    Mat channel = new Mat();
                    Core.extractChannel(img, channel, k);
                    Mat enhancedChannel = ALTM(channel);
                    channels.set(k, enhancedChannel);

               }
                // 合并通道
                Core.merge(channels, res);
                res.convertTo(res, CvType.CV_8UC3);
                Bitmap resultBitmap = Bitmap.createBitmap(res.cols(), res.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(res, resultBitmap);
                resultImageView.setImageBitmap(resultBitmap);




            }
        });
    }
    public static Mat ALTM(Mat img) {
        Mat res = new Mat();
        int h = img.rows();
        int w = img.cols();
        double Lwmax = Core.minMaxLoc(img).maxVal;
        System.out.print(Lwmax);
        Mat log_Lw = new Mat();

        Core.add(img, Scalar.all(0.001), img);
        Core.log(img, log_Lw);




        double Lw_sum = Core.sumElems(log_Lw).val[0];
        double Lwaver = Math.exp(Lw_sum / (h * w));

        Mat Lg = new Mat();
        Core.divide(img, new Scalar(Lwaver), Lg);
        Core.add(Lg, Scalar.all(1), Lg);
        Core.log(Lg, Lg);
        Core.divide(Lg, new Scalar(Math.log(Lwmax / Lwaver) + 1), Lg);

        Core.multiply(Lg, new Scalar(255.0), Lg);
        Lg.convertTo(res, CvType.CV_8UC1);

        return res;
    }

    private void initLoadOpenCv(){
        boolean success= OpenCVLoader.initDebug();
        if(success){
            Toast.makeText(this.getApplicationContext(), "Loading Opencv Libraries", Toast.LENGTH_SHORT).show();
        }
        else{
            Toast.makeText(this.getApplicationContext(), "WARNING：COULD NOT LOAD Opencv Libraries!", Toast.LENGTH_SHORT).show();
        }
    }
}