using System.Configuration;
using System.Data;
using System.Windows;

namespace OpenCV
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        protected override void OnStartup(StartupEventArgs e)
        {
            base.OnStartup(e);

            // WPF 창 실행
            MainWindow mainWindow = new MainWindow();
            CameraDetection.MainWin = mainWindow;
            mainWindow.Show();

            // 별도 스레드로 CameraDetection 실행
            Thread cameraThread = new Thread(() =>
            {
                // 카메라 창 실행
                CameraDetection.CamerDetectionDemo();
            });
            cameraThread.IsBackground = true;
            cameraThread.Start();
        }
    }

}
