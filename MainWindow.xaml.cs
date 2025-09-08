using System.Collections.ObjectModel;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace OpenCV
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public ObservableCollection<HelmetStatus> StatusList { get; set; }

        public MainWindow()
        {
            InitializeComponent();

            StatusList = new ObservableCollection<HelmetStatus>();
            DataGridStatus.ItemsSource = StatusList;
        }

        // DataGrid에 표시할 데이터 구조
        public class HelmetStatus
        {
            public int Idx { get; set; }
            public string Status { get; set; }
            public string Time { get; set; }
        }

        public void AddStatus(int index, string status, DateTime time)
        {
            // UI 스레드 안전하게 업데이트
            Dispatcher.Invoke(() =>
            {
                StatusList.Add(new HelmetStatus
                {
                    Idx = index,
                    Status = status,
                    Time = time.ToString("yyyy-MM-dd HH:mm:ss")
                });
            });
        }
    }
}