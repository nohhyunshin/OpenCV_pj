using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

// YOLO
namespace project2
{
    public static class ImageClassifier
    {
        private static readonly string[] ClassNames =
        {
            "crazing",
            "inclusion",
            "patches",
            "pitted_surface",
            "rolled_in_scale",
            "scratches"
        };

        private static InferenceSession? _session;
        private static string _inputName = "images";

        private static InferenceSession GetSession()
        {
            if (_session != null) return _session;

            string onnxPath = "best (1).onnx";
            // string onnxPath = "detector.onnx";
            if (!File.Exists(onnxPath))
                throw new FileNotFoundException($"ONNX model not found: {Path.GetFullPath(onnxPath)}");

            var so = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(onnxPath, so);

            var shape = _session.OutputMetadata["output0"].Dimensions; 
            Debug.WriteLine(string.Join(",", shape)); // 1, 10, 1029

            Debug.WriteLine("== ONNX Output Metadata ==");
            foreach (var kv in _session.OutputMetadata)
                Debug.WriteLine($"OUT {kv.Key} : {kv.Value.ElementType} {string.Join('x', kv.Value.Dimensions)}");

            return _session;
        }

        public static Dictionary<string, float> ClassifyImage(string imagePath, bool showAnnotatedWindow = false, float scoreThresh = 0.3f)
        {
            using var img = Cv2.ImRead(imagePath);
            if (img.Empty()) throw new Exception("이미지 로드 실패");

            var session = GetSession();

            int inputWidth = 224;
            int inputHeight = 224;
            // YOLO 입력 크기

            // 이미지 리사이즈
            using var resized = new Mat();
            Cv2.Resize(img, resized, new Size(inputWidth, inputHeight));

            // OpenCV DNN 전처리
            using var blob = CvDnn.BlobFromImage(resized, 1.0 / 255.0, new Size(inputWidth, inputHeight),
                                                 new Scalar(0, 0, 0), swapRB: true, crop: false);

            // DenseTensor 생성
            var tensor = new DenseTensor<float>(new[] { 1, 3, inputHeight, inputWidth });
            float[] tmp = new float[1 * 3 * inputHeight * inputWidth];
            Marshal.Copy(blob.Data, tmp, 0, tmp.Length);
            tmp.CopyTo(tensor.Buffer.Span);

            // 모델 실행
            using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });

            // 결과가 없으면 빈 딕셔너리 반환
            if (!results.Any())
                return new Dictionary<string, float>();

            // YOLO 출력 텐서 (Flattened 배열로 가정)
            var outputTensor = results.First(r => r.Name == "output0").AsTensor<float>();
            var output = outputTensor.ToArray(); // 1D 배열
            int rowSize = outputTensor.Dimensions[1]; // x,y,w,h,conf + class scores
            int numClasses = 6;
            int numDetections = 5 + numClasses;

            float[] classMax = new float[numClasses];

            for (int i = 0; i < numDetections; i++)
            {
                float conf = output[4 * numDetections + i]; // objectness
                float[] classScores = output.Skip(i * rowSize + 5).Take(numClasses).ToArray();

                for (int c = 0; c < classScores.Length; c++)
                {
                    Debug.WriteLine($"Class {c} ({ClassNames[c]}): {classScores[c]}");
                }

                int cls = Array.IndexOf(classScores, classScores.Max());
                float finalScore = conf * classScores[cls];

                if (finalScore < scoreThresh || cls < 0 || cls >= numClasses) continue;
                if (finalScore > classMax[cls]) classMax[cls] = finalScore;
            }

            float sum = classMax.Sum();
            var probs = new Dictionary<string, float>();

            for (int c = 0; c < numClasses; c++)
                probs[ClassNames[c]] = sum > 0 ? classMax[c] / sum : 0;

            return probs;
        }
    }
}
