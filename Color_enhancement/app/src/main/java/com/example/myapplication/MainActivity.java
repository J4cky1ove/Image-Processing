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
import org.opencv.core.MatOfFloat;
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
    public static double eps = Double.MIN_VALUE;
    @Override
    protected void onDestroy() {
        super.onDestroy();
        mat1.release();
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initLoadOpenCv();
        bt1=findViewById(R.id.button);

        System.out.println("Function has generated zero.");

        bt1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Get drawable resource

                ImageView resultImageView = findViewById(R.id.imageButton);
                Context context = getApplicationContext();
                Resources resources = context.getResources();
                @SuppressLint("ResourceType") InputStream inputStream = resources.openRawResource(R.drawable.p1);
                File outputFile = new File(getFilesDir(), "p1.png");
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
                img.convertTo(img, CvType.CV_32FC3);
                long startTime = System.currentTimeMillis();
                Mat res = new Mat(img.size(), CvType.CV_32FC3);
                // 设置通道的大小
                List<Mat> channels = new ArrayList<>();
                Size channelSize =  img.size();
                double[] sigmas = {12,80,250};
                res=retinexMSRCP(img, sigmas, 0.01, 0.01);
                Bitmap resultBitmap = Bitmap.createBitmap(res.cols(), res.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(res, resultBitmap);
                resultImageView.setImageBitmap(resultBitmap);
                long endTime = System.currentTimeMillis();
                long runningTime = endTime - startTime;
                System.out.println("App running time is " + runningTime + " millisecond");
            }
        });
    }
    public static Mat ALTM(Mat img) {
        Mat res = new Mat();
        int h = img.rows();
        int w = img.cols();
        double Lwmax = Core.minMaxLoc(img).maxVal;
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
    public static Mat getGaussKernel(double sigma, int dim) {
        int ksize = (int) (sigma * 2) * 2 + 1;
        Mat x = new Mat(1, ksize, CvType.CV_64F);
        Mat y = new Mat(ksize, 1, CvType.CV_64F);
        for (int i = 0; i < ksize; i++) {
            int halfK = ksize / 2;
            x.put(0, i, i - halfK);
            y.put(i, 0, i - halfK);
        }
        Mat kernel = new Mat(ksize, ksize, CvType.CV_64F);
        Mat xSquare = new Mat(), ySquare = new Mat();
        Core.pow(x, 2, xSquare); // square of x
        Core.pow(y, 2, ySquare); // square of y
        Core.multiply(xSquare, new Scalar(0.5 / (sigma * sigma)), xSquare); // scale xSquare
        Core.multiply(ySquare, new Scalar(0.5 / (sigma * sigma)), ySquare); // scale ySquare
        Core.add(xSquare, ySquare, kernel); // add xSquare and ySquare
        Core.multiply(kernel, new Scalar(-1), kernel); // multiply with -1
        Core.exp(kernel, kernel); // apply exp
        Core.divide(kernel, new Scalar(Core.sumElems(kernel).val[0]), kernel);
        if (dim == 1) {
            return kernel;
        } else if (dim == 2) {
            Mat kernel2D = new Mat();
            Core.gemm(kernel.t(), kernel, 1, new Mat(), 0, kernel2D, 0);  // Use flag as 0
            return kernel2D;
        }
        return null;
    }

    public static Mat gaussBlur(Mat img, double sigma) {
        int ksize = 7;
        Mat blurred = new Mat();
        Imgproc.GaussianBlur(img, blurred, new Size(ksize, ksize), sigma);
        return blurred;
    }

    public static Mat simplestColorBalance(Mat imgMsrcr, double s1, double s2) {
        Mat sortedImg = new Mat();
        Core.sort(imgMsrcr.reshape(1, 1), sortedImg, Core.SORT_ASCENDING);
        int N = imgMsrcr.rows() * imgMsrcr.cols();
        int vminIndex = (int) (N * s1);
        int vmaxIndex = (int) (N * (1 - s2)) - 1;
        double Vmin = sortedImg.get(0, vminIndex)[0];
        double Vmax = sortedImg.get(0, vmaxIndex)[0];

        // Clip the imgMsrcr
        Mat lowerThanVmin = new Mat();
        Mat higherThanVmax = new Mat();

        Core.compare(imgMsrcr, new Scalar(Vmin), lowerThanVmin, Core.CMP_LT); // pixels < Vmin
        Core.compare(imgMsrcr, new Scalar(Vmax), higherThanVmax, Core.CMP_GT); // pixels > Vmax

        imgMsrcr.setTo(new Scalar(Vmin), lowerThanVmin);
        imgMsrcr.setTo(new Scalar(Vmax), higherThanVmax);

        Mat clamped = new Mat();
        Core.subtract(imgMsrcr, Scalar.all(Vmin), clamped);
        Core.multiply(clamped, new Scalar(255 / (Vmax - Vmin)), clamped);
        Mat balanced = new Mat();
        clamped.convertTo(balanced, CvType.CV_8UC1);

        return balanced;
    }


    public static <MatVector> Mat retinexMSRCP(Mat img, double[] sigmas, double s1, double s2) {
        Mat intImg = new Mat();
        List<Mat> channels = new ArrayList<>();
        Core.split(img, channels);

        // Access each channel
        Mat blueChannel = channels.get(0);
        Mat greenChannel = channels.get(1);
        Mat redChannel = channels.get(2);

        //Int=np.sum(img,axis=2)/3
        //---------------------------------------------------------------
        Core.add(blueChannel, greenChannel, intImg);
        Core.add(intImg, redChannel, intImg);

        Core.divide(intImg, new org.opencv.core.Scalar(3), intImg);
        intImg.convertTo(intImg, CvType.CV_64F);

        //---------------------------------------------------------------

        // for sigma in sigmas:
        Mat[] diffs = new Mat[sigmas.length];
        for (int i = 0; i < sigmas.length; i++) {
            //        Diffs.append(np.log(Int+1)-np.log(gauss_blur(Int,sigma)+1))
            // ---------------------------------------------------------------------------
            Mat diff = new Mat();

            // gauss_blur(Int,sigma)
            Mat blurred = gaussBlur(intImg, sigmas[i]);
            // gauss_blur(Int,sigma) + 1
            int rows = blurred.rows();
            int cols = blurred.cols();

            Core.add(blurred, new org.opencv.core.Scalar(1), blurred);
            // np.log(gauss_blur(Int,sigma)+1)
            Core.log(blurred, blurred);
            // Int + 1
            Mat int_img_1 = new Mat();
            Core.add(intImg, new org.opencv.core.Scalar(1), int_img_1);
            // np.log(Int+1)
            Core.log(int_img_1, int_img_1);
            // np.log(Int+1)-np.log(gauss_blur(Int,sigma)+1)
            Core.subtract(int_img_1, blurred, diff);
            diffs[i] = diff;
            // ---------------------------------------------------------------------------
        }

        // MSR=sum(Diffs)/3
        // ---------------------------------------------------------------------------
        Mat msr = new Mat();
        Mat temp = new Mat();
        Core.add(diffs[0], diffs[1], temp);
        for (int i = 0; i < sigmas.length; i++) {
            if (i == 0 || i == 1){
                continue;
            }
            else{
                Core.add(temp, diffs[i], msr);
            }
        }
        temp.release();
        Core.divide(msr, new org.opencv.core.Scalar(3), msr);
        // ---------------------------------------------------------------------------
        Mat int1 = simplestColorBalance(msr, s1, s2);
        // B=np.max(img,axis=2)
        // ---------------------------------------------------------------------------
        Mat b = new Mat();

        Core.max(greenChannel, redChannel, b);
        Core.max(b, blueChannel, b);
        // ---------------------------------------------------------------------------

        // 255/(B+eps)
        // Int1/(Int+eps)
        // ---------------------------------------------------------------------------
        Mat a = new Mat();
        // (B + eps)
        Mat b_eps = new Mat();
        Core.add(b, new org.opencv.core.Scalar(eps), b_eps);

        // b_eps = 255/(B+eps)
        Mat mat255 = new Mat(b_eps.size(), CvType.CV_32FC1, new org.opencv.core.Scalar(255));

        Core.divide(mat255, b_eps, b_eps);
        // Int+eps
        Mat intImg_eps = new Mat();
        Core.add(intImg, new Scalar(eps), intImg_eps);
        intImg_eps.convertTo(intImg_eps, CvType.CV_32FC1);
        int1.convertTo(int1, CvType.CV_32FC1);
        Core.divide(int1, intImg_eps, intImg_eps); // intImg_eps = Int1/(Int+eps)
        // ---------------------------------------------------------------------------

        //np.stack()
        // ---------------------------------------------------------------------------
        List<Mat> mats = new ArrayList<>();
        mats.add(intImg_eps);
        mats.add(b_eps);
        Mat stack = new Mat();
        Core.merge(mats, stack);
        // ---------------------------------------------------------------------------
        //np.min
        // ---------------------------------------------------------------------------
        List<Mat> findMin = new ArrayList<>();
        Core.split(stack, findMin);
        Mat final_min = new Mat();
        Core.min(findMin.get(0), findMin.get(1), final_min);
        // ---------------------------------------------------------------------------
        //A[...,None]
        // ---------------------------------------------------------------------------
        List<Mat> addDim = new ArrayList<>();
        addDim.add(final_min);
//        Mat ret = new Mat();
//        Core.merge(addDim, ret);
        // ---------------------------------------------------------------------------

        //A[...,None]*img
        // ---------------------------------------------------------------------------
        Mat result = new Mat();
        Mat redResult = new Mat();
        Mat blueResult = new Mat();
        Mat greeResult = new Mat();
        Core.multiply(final_min, redChannel, redResult);
        Core.multiply(final_min, blueChannel, blueResult);
        Core.multiply(final_min, greenChannel, greeResult);
        List<Mat> mergeColor = new ArrayList<>();
        mergeColor.add(redResult);
        mergeColor.add(greeResult);
        mergeColor.add(blueResult);
        Core.merge(mergeColor, result);
        List<Mat> test = new ArrayList<>();

        result.convertTo(result, CvType.CV_8UC1);
        // ---------------------------------------------------------------------------
        return result;
    }

    public static void printMat(Mat mat){
        int rows = mat.rows();
        int cols = mat.cols();

        System.out.println("your mat rows " + rows);
        System.out.println("your mat cols " + cols);

        System.out.println("your mat is ");

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double value = mat.get(i, j)[0];
                System.out.print(value + " ");
            }
            System.out.println();
        }
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