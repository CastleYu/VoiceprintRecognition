# Database配置

## mysql

- 启动：docker run --privileged=true  -v .\data\:/var/lib/mysql -v .\logs\:/var/log/mysql -v .\conf\:/etc/mysql -v .\my.cnf:/etc/mysql/my.cnf  -p 8886:3306 --name mysql -e MYSQL_ROOT_PASSWORD=123456 -d mysql（其中.\data目录在我github仓库项目中的database\mysql目录下，可以根据需要挂载到你自己的地址）

- 可视化：Navicat等工具





## milvus

- 启动：首先切换到目录Speech\database\milvus下，可以看到一个docker-compose.yml文件，docker compose up -d即可
- 可视化：https://github.com/zilliztech/attu