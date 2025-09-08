using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// Faster R-CNN
namespace project2
{
    internal class ImageClassifier2
    {
        private static readonly string[] ClassNames =
        {
            "background",
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

            string onnxPath = "detector.onnx";
            if (!File.Exists(onnxPath))
                throw new FileNotFoundException($"ONNX model not found: {Path.GetFullPath(onnxPath)}");

            var so = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            _session = new InferenceSession(onnxPath, so);

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

            // OpenCV DNN으로 전처리
            using var blob = CvDnn.BlobFromImage(img, 1.0 / 255.0, new Size(img.Cols, img.Rows),
                                                 new Scalar(0, 0, 0), swapRB: true, crop: false);

            var tensor = new DenseTensor<float>(new[] { 1, 3, img.Rows, img.Cols });
            var tmp = new float[(int)blob.Total()];
            System.Runtime.InteropServices.Marshal.Copy(blob.Data, tmp, 0, tmp.Length);
            tmp.CopyTo(tensor.Buffer.Span);

            using var results = session.Run(new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) });
            var map = results.ToDictionary(r => r.Name, r => r);

            var scoresAny = map.ContainsKey("scores") ? map["scores"] : results.ElementAt(1);
            var labelsAny = map.ContainsKey("labels") ? map["labels"] : results.ElementAt(2);

            var scoresT = scoresAny.AsTensor<float>();
            int n = scoresT.Dimensions[0];

            int[] labels;
            try
            {
                labels = labelsAny.AsTensor<long>().Select(l => (int)l).ToArray();
            }
            catch
            {
                labels = labelsAny.AsTensor<float>().Select(l => (int)Math.Round(l)).ToArray();
            }

            float[] classMax = new float[ClassNames.Length];

            for (int i = 0; i < n; i++)
            {
                float score = scoresT[i];
                int cls = labels[i];
                if (score < scoreThresh || cls <= 0 || cls >= ClassNames.Length) continue;
                if (score > classMax[cls]) classMax[cls] = score;
            }

            float sum = classMax.Skip(1).Sum();
            var probs = new Dictionary<string, float>();

            for (int c = 1; c < ClassNames.Length; c++)
                probs[ClassNames[c]] = sum > 0 ? classMax[c] / sum : 0;

            return probs;
        }
    }
}
