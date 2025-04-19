#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>

int main() {
    cv::VideoCapture cap("../../res/video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: Не удалось открыть видеофайл!" << std::endl;
        return -1;
    }

    // Загружаем каскады Хаара
    cv::CascadeClassifier face_cascade, eye_cascade, smile_cascade;
    if (!face_cascade.load("haarcascade_frontalface_default.xml")) {
        std::cerr << "Ошибка: Не удалось загрузить каскад для лиц!" << std::endl;
        return -1;
    }
    if (!eye_cascade.load("haarcascade_eye.xml")) {
        std::cerr << "Ошибка: Не удалось загрузить каскад для глаз!" << std::endl;
        return -1;
    }
    if (!smile_cascade.load("haarcascade_smile.xml")) {
        std::cerr << "Ошибка: Не удалось загрузить каскад для улыбок!" << std::endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter video_writer("../../out/output_video.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(frame_width, frame_height));
    if (!video_writer.isOpened()) {
        std::cerr << "Ошибка: Не удалось создать файл для записи видео!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Ошибка: Кадр пустой!" << std::endl;
            break;
        }
        frames.push_back(frame.clone());
    }
    cap.release();

    int num_threads = omp_get_max_threads();
    std::vector<cv::CascadeClassifier> face_cascades(num_threads);
    std::vector<cv::CascadeClassifier> eye_cascades(num_threads);
    std::vector<cv::CascadeClassifier> smile_cascades(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        if (!face_cascades[i].load("haarcascade_frontalface_default.xml")) {
            std::cerr << "Ошибка: Не удалось загрузить каскад для лиц в потоке " << i << "!" << std::endl;
            return -1;
        }
        if (!eye_cascades[i].load("haarcascade_eye.xml")) {
            std::cerr << "Ошибка: Не удалось загрузить каскад для глаз в потоке " << i << "!" << std::endl;
            return -1;
        }
        if (!smile_cascades[i].load("haarcascade_smile.xml")) {
            std::cerr << "Ошибка: Не удалось загрузить каскад для улыбок в потоке " << i << "!" << std::endl;
            return -1;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Параллельная обработка кадров
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < frames.size(); ++i) {
        int thread_id = omp_get_thread_num();

        cv::Mat& frame = frames[i];

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Детекция лиц
        std::vector<cv::Rect> faces;
        face_cascades[thread_id].detectMultiScale(gray, faces, 1.1, 3, 0, cv::Size(30, 30));

        // Для каждого найденного лица
        for (size_t j = 0; j < faces.size(); ++j) {
            #pragma omp critical
            cv::rectangle(frame, faces[j], cv::Scalar(255, 0, 0), 2);

            cv::Mat faceROI = gray(faces[j]);

            // Детекция глаз
            std::vector<cv::Rect> eyes;
            eye_cascades[thread_id].detectMultiScale(faceROI, eyes, 1.5, 5, 0, cv::Size(20, 20), cv::Size(50, 50));
            for (size_t k = 0; k < eyes.size(); ++k) {
                cv::Point eye_center(faces[j].x + eyes[k].x + eyes[k].width / 2,
                                     faces[j].y + eyes[k].y + eyes[k].height / 2);
                int radius = cvRound((eyes[k].width + eyes[k].height) * 0.25);
                #pragma omp critical
                cv::rectangle(frame,
                              cv::Point(faces[j].x + eyes[k].x, faces[j].y + eyes[k].y),
                              cv::Point(faces[j].x + eyes[k].x + eyes[k].width, faces[j].y + eyes[k].y + eyes[k].height),
                              cv::Scalar(0, 0, 255), 2);
            }

            // Детекция улыбки
            std::vector<cv::Rect> smiles;
            smile_cascades[thread_id].detectMultiScale(faceROI, smiles, 1.5, 20, 0, cv::Size(20, 20));
            for (size_t k = 0; k < smiles.size(); ++k) {
                #pragma omp critical
                cv::rectangle(frame,
                              cv::Point(faces[j].x + smiles[k].x, faces[j].y + smiles[k].y),
                              cv::Point(faces[j].x + smiles[k].x + smiles[k].width, faces[j].y + smiles[k].y + smiles[k].height),
                              cv::Scalar(0, 255, 0), 2);
            }
        }

        // Записываем кадр в выходное видео
        #pragma omp critical
        video_writer.write(frame);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "Duration of video processing with multithreading: " << duration.count() / 1000.0 << " sec" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

    // Отображаем кадры
    for (const auto& frame : frames) {
        cv::imshow("Обнаружение лиц, глаз и улыбок", frame);
        if (cv::waitKey(30) == 'q') {
            break;
        }
    }

    // Освобождаем ресурсы
    video_writer.release();
    cv::destroyAllWindows();
    return 0;
}
