Данный проект - задание полученное на производственной практике.
Проект решает задачу распознования на фотографиях(RGB-изображениях) “точек-интереса”. 
В качестве “точек интереса” (Point-of-Interest) выступают углы прямоугольников (или близких к ним фигур), присутствующих на изображениях. 
Прямоугольники образуются гранями параллелепипедов перпендикулярные (или “почти перпендикулярные”) к оптической оси камеры. 

Postman-коллекция находится в файле BoxCornerIdentifier API.postman_collection.json
Swagger-документация представленна в виде schema.yaml

Для запуска проекта необходимо использовать Docker, для запуска необходимо использовать комманду:
docker-compose up

При создании образа создается суперпользователь, где:
имя пользователся/username: admin
пароль/password: admin
