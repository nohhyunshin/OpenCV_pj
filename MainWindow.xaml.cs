using Microsoft.Win32;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using static System.Reflection.Metadata.BlobBuilder;

namespace project2
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private string? _selectedImage;

        public MainWindow()
        {
            InitializeComponent();

            // WPF 좌상단 현재 시간 출력
            var timer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
            timer.Tick += (s, e) => TodayText.Text = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            timer.Start();
        }

        // Input 버튼
        private void BtnInput_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "이미지 파일|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"
            };
            if (dlg.ShowDialog() == true)
            {
                _selectedImage = dlg.FileName;

                // 이미지 미리보기
                var bitmap = new BitmapImage();
                bitmap.BeginInit();
                bitmap.UriSource = new Uri(_selectedImage);
                bitmap.CacheOption = BitmapCacheOption.OnLoad;
                bitmap.EndInit();
                validationImg.Source = bitmap;

                // 다시 이미지 넣을 때 출력된 값 초기화
                ResultGrid.ItemsSource = null;
                AnalysisTimeText.Text = $"Run Time (s) : ";
                FileText.Text = $"Source : ";
                AnalysisDefectText.Text = $"Analysis : ";
            }
        }

        // Output 버튼
        private void BtnOutput_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedImage) || !File.Exists(_selectedImage))
            {
                MessageBox.Show("분석할 이미지를 선택하세요.");
                return;
            }

            try
            {
                var sw = Stopwatch.StartNew();

                // var probs = ImageClassifier.ClassifyImage(_selectedImage, showAnnotatedWindow: false);
                var probs = ImageClassifier2.ClassifyImage(_selectedImage, showAnnotatedWindow: false);

                sw.Stop();

                // Datagrid에 출력
                ResultGrid.ItemsSource = probs
                    .OrderByDescending(kv => kv.Value)
                    .Select(kv => new { Key = kv.Key, Value = (kv.Value * 100).ToString("F2") });

                // 러닝 타임 표시
                AnalysisTimeText.Text = $"Run Time (s) : {sw.Elapsed.TotalSeconds:F2}s";

                // 원본 파일 이름 표시
                string fileName = System.IO.Path.GetFileName(_selectedImage);
                FileText.Text = $"Source : {fileName}";

                // 분석 결과 표시
                var top = probs.OrderByDescending(kv => kv.Value).First();
                AnalysisDefectText.Text = $"Analysis : {top.Key} ({(top.Value * 100):F2}%)";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"분류 중 오류 : {ex.Message}");
            }
        }
    }
}