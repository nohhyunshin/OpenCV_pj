using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using Point = OpenCvSharp.Point;
using Rect = OpenCvSharp.Rect;
using Size = OpenCvSharp.Size;

using project2;

namespace OpenCV
{
    internal class CameraDetection
    {
        public static MainWindow MainWin { get; set; }

        // 카운트 누적을 위함 (24시간에 한 번씩 초기화)
        private static Dictionary<int, int> objectCountDict = new();
        private static System.Timers.Timer dailyResetTimer;     // 타이머 초기화

        // 메세지 박스 자주 뜨는 것 방지 (쿨다운)
        private static DateTime lastWarningTime = DateTime.MinValue;
        private static readonly TimeSpan warningCooldown = TimeSpan.FromSeconds(3);

        // 출력값
        private static int displayValue = 1;
        private static int indexValue = 0;

        // project2 실행 여부 플래그
        private static bool project2Launched = false;
        private static bool cameraPaused = false;   // project2 실행 시 카메라 일시 정지

        public static void CamerDetectionDemo()
        {
            // 24시간마다 displayValue 초기화
            dailyResetTimer = new System.Timers.Timer(24 * 60 * 60 * 1000); // 24시간
            dailyResetTimer.Elapsed += (s, e) => displayValue = 0;
            dailyResetTimer.Start();

            int frameCount = 0;

            string modelPath = "helmet.onnx";

            var session = new InferenceSession(modelPath);

            // ONNX 출력 이름 확인
            //Debug.WriteLine("=== ONNX Model Outputs ===");
            //foreach (var o in session.OutputMetadata)
            //{
            //    Debug.WriteLine($"Output: {o.Key}  ->  {o.Value.ElementType} [{string.Join(",", o.Value.Dimensions)}]");
            //}
            //Debug.WriteLine("==========================");

            // 카메라 열기
            VideoCapture cam = new VideoCapture(0);
            if (!cam.IsOpened()) return;

            Mat frame = new Mat();
            while (true)
            {
                cam.Read(frame);
                if (frame.Empty()) return;

                // 밝기, 대비 조정
                Mat adjusted = new Mat();
                Cv2.CvtColor(frame, adjusted, ColorConversionCodes.BGR2YCrCb);

                var channels = Cv2.Split(adjusted);
                var clahe = Cv2.CreateCLAHE(clipLimit: 2.0, tileGridSize: new Size(8, 8));
                clahe.Apply(channels[0], channels[0]);  // Y 채널만 CLAHE 적용

                Cv2.Merge(channels, adjusted);
                Cv2.CvtColor(adjusted, adjusted, ColorConversionCodes.YCrCb2BGR);

                // GaussianBlur로 노이즈 제거
                Cv2.GaussianBlur(frame, frame, new Size(3, 3), 0);

                // Tensor 전처리
                Mat resized = new Mat();
                Cv2.CvtColor(frame, resized, ColorConversionCodes.BGR2RGB);

                // !! 모델 변환할 때 dummy input에 맞춰서 resize !! 확인 필요
                // 학습은 416로 했지만, 추론 시 416로 resize하면 오히려 detection score가 높을 수 있음
                // 실무에서는 학습 크기와 상관없이 실제 inference 환경에서 가장 잘 잡히는 resize 크기를 사용하는 게 안전
                // > 416로 조정한 이유
                Cv2.Resize(resized, resized, new Size(416, 416));

                var inputTensor = new DenseTensor<float>(new[] { 1, 3, 416, 416 });
                var data = new float[3 * 416 * 416];
                int idx = 0;
                for (int c = 0; c < 3; c++)
                {
                    for (int y = 0; y < 416; y++)
                    {
                        for (int x = 0; x < 416; x++)
                        {
                            data[idx++] = resized.At<Vec3b>(y, x)[c] / 255.0f;
                        }
                    }
                }
                data.CopyTo(inputTensor.Buffer.Span);

                // 추론 실행
                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("images", inputTensor)
                };
                using var results = session.Run(inputs);

                // output0 하나만 존재
                var output = results.First().AsEnumerable<float>().ToArray();

                // output 구조: [x1, y1, x2, y2, score, label] × num_boxes
                // 0: 'helmet', 1: 'person', 2: 'head'으로 학습된 모델
                int numBoxes = output.Length / 6;

                // 사람, 헬멧, 머리 박스 리스트
                var persons = new List<(int x1, int y1, int x2, int y2, int label, float score)>();
                var helmets = new List<(int x1, int y1, int x2, int y2)>();
                var heads = new List<(int x1, int y1, int x2, int y2)>();

                // 뭐로 인식되는지 확인
                //var labelSet = new HashSet<int>();
                //for (int i = 0; i < numBoxes; i++)
                //{
                //    int label = (int)output[i * 6 + 5];
                //    labelSet.Add(label);
                //}
                //Debug.WriteLine("Detected labels: " + string.Join(", ", labelSet));

                for (int i = 0; i < numBoxes; i++)
                {
                    // output 배열에서 값 추출
                    float score = output[i * 6 + 4];
                    if (score < 0.5f) continue; // confidence threshold

                    int x1 = (int)output[i * 6];
                    int y1 = (int)output[i * 6 + 1];
                    int x2 = (int)output[i * 6 + 2];
                    int y2 = (int)output[i * 6 + 3];
                    int label = (int)output[i * 6 + 5]; // class id

                    // 좌표를 원본 프레임 크기에 맞게 스케일링
                    float scaleX = (float)frame.Width / 416f;
                    float scaleY = (float)frame.Height / 416f;

                    x1 = (int)(x1 * scaleX);
                    y1 = (int)(y1 * scaleY);
                    x2 = (int)(x2 * scaleX);
                    y2 = (int)(y2 * scaleY);

                    // 리스트에 분류
                    if (label == 1) persons.Add((x1, y1, x2, y2, label, score));
                    else if (label == 0) helmets.Add((x1, y1, x2, y2));
                    else if (label == 2) heads.Add((x1, y1, x2, y2));
                }

                // 사람인지 판단 후 헬멧인지 머리인지 판단
                // 사람 + 헬멧 : royalblue
                // 사람 + 머리 : magenta
                foreach (var person in persons)
                {
                    bool hasHelmet = helmets.Any(h => RectOverlap(person.x1, person.y1, person.x2, person.y2,
                                                                  h.x1, h.y1, h.x2, h.y2));
                    bool hasHead = heads.Any(h => RectOverlap(person.x1, person.y1, person.x2, person.y2,
                                                              h.x1, h.y1, h.x2, h.y2));

                    Scalar boxColor;
                    string labelText;

                    // 헬멧 착용 시 WPF + ImageClassifier 실행
                    if (hasHelmet && !project2Launched)
                    {
                        boxColor = Scalar.RoyalBlue;
                        labelText = $"Worn {(person.score * 100):F2}%";

                        project2Launched = true;
                        cameraPaused = true;

                        MainWin?.Dispatcher.BeginInvoke(() =>
                        {
                            project2.MainWindow win = new project2.MainWindow();

                            win.Closed += (s, e) =>
                            {
                                cameraPaused = false;   // 카메라 재개
                                project2Launched = false;
                            };

                            win.Show();
                        });
                    }
                    else if (hasHead)
                    {
                        boxColor = Scalar.Magenta;
                        labelText = $"Not Worn {(person.score*100):F2}%";
                    }
                    else
                    {
                        continue; // 단순히 사람만 있으면 아무것도 표시 x
                    }

                    // WPF로 전달
                    string label = hasHelmet ? "착용" : "미착용";
                    MainWin?.Dispatcher.BeginInvoke(() =>
                    {
                        MainWin.AddStatus(displayValue++, label, DateTime.Now);
                    });

                    // 객체 윤곽선을 얼굴 영역 정도로 축소
                    int boxWidth = person.x2 - person.x1;
                    int boxHeight = person.y2 - person.y1;

                    float faceScale = 0.5f; // 전체 바운딩 박스의 40% 정도만 사용
                    int newWidth = (int)(boxWidth * faceScale);
                    int newHeight = (int)(boxHeight * faceScale) + 20;
                    int newX = person.x1 + (boxWidth - newWidth) / 2;  // 중앙 정렬
                    int newY = person.y1; // 상단 기준

                    Cv2.Rectangle(frame, new Rect(newX, newY + 30, labelText.Length * 10, 20), boxColor, -1);    // 라벨 박스
                    Cv2.Rectangle(frame, new Rect(newX, newY + 50, newWidth, newHeight), boxColor, 2);
                    Cv2.PutText(frame, labelText, new OpenCvSharp.Point(newX + 3, newY + 45),
                                HersheyFonts.HersheyComplex, 0.5, Scalar.White, 1);
                }
                // 현재 날짜 및 시간 정보 (우상단)
                AddVideoInfo(frame, DateTime.Now);

                // 헬멧 미착용 여부 판단
                bool anyNoHelmet = persons.Any(p =>
                {
                    bool hasHelmet = helmets.Any(h => RectOverlap(p.x1, p.y1, p.x2, p.y2, h.x1, h.y1, h.x2, h.y2));
                    bool hasHead = heads.Any(h => RectOverlap(p.x1, p.y1, p.x2, p.y2, h.x1, h.y1, h.x2, h.y2));
                    return hasHead && !hasHelmet;   // 사람 + 머리만 있으면 미착용으로 판단
                });


                // 미착용 시 캡처
                if (anyNoHelmet && DateTime.Now - lastWarningTime > warningCooldown)
                {
                    // lastWarningTime = DateTime.Now;     // 쿨다운
                    ErrorCapture(frame, DateTime.Now, false);
                }

                // 카메라 화면
                Cv2.ImShow("Helmet Usage Status", frame);

                int key = Cv2.WaitKey(30);
                if (key == 27) break;
            }
            cam.Release();      // 카메라 장치 닫기
            Cv2.DestroyAllWindows();
        }

        // 사각형 겹침 여부 판단 함수
        private static bool RectOverlap(int x1a, int y1a, int x2a, int y2a,
                                        int x1b, int y1b, int x2b, int y2b)
        {
            int x_overlap = Math.Max(0, Math.Min(x2a, x2b) - Math.Max(x1a, x1b));
            int y_overlap = Math.Max(0, Math.Min(y2a, y2b) - Math.Max(y1a, y1b));
            return x_overlap > 0 && y_overlap > 0;
        }
            
        // 카메라 화면 정보 출력
        private static void AddVideoInfo(Mat video, DateTime now)
        {
            // 시간 정보
            string txtTime = $"{now:yyyy-MM-dd HH:mm:ss}";
            Cv2.PutText(video, txtTime, new Point(440, 25), HersheyFonts.HersheyComplex, 0.5, Scalar.Tomato);
        }

        // 에러 발생 시 캡처
        private static void ErrorCapture(Mat video, DateTime now, bool hasHelmet)
        {
            // 저장 폴더 경로
            string folder = @"C:\Users\user\Desktop\OpenCV";
            if (!System.IO.Directory.Exists(folder))
            {
                // 폴더 없으면 생성
                System.IO.Directory.CreateDirectory(folder);
            }

            // 헬멧 미착용 시 캡처
            string filename = System.IO.Path.Combine(folder, $"Helmet Now Worn_{now:yyyy-MM-dd}_{displayValue}.jpg");

            try
            {
                if (!hasHelmet)
                {
                    Cv2.ImWrite(filename, video);
                    MessageBox.Show("안전모를 착용해 주세요.", "경고", MessageBoxButton.OK, MessageBoxImage.Warning);
                    Cv2.WaitKey(1000);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"예외 발생 : {e.Message}");
            }
        }
    }
}