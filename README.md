Xây dựng Docker image:

```bash
docker-compose up
```

```bash
docker build -t translator-api .
```

Chạy ứng dụng với Docker:

```bash
docker run -p 5001:5000 translator-api
```

Hoặc nếu sử dụng Docker Compose:

```bash
docker-compose up --build
```
