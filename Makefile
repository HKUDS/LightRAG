# 定义变量
IMAGE_NAME ?= lightrag-api
CONTAINER_NAME ?= lightrag-api
IMAGE_VERISON ?= $(shell git rev-parse --short HEAD)
REGISTRY_URL ?= 192.168.0.24:7001
PLATFORMS ?= linux/amd64,linux/arm64

# 检查命令执行状态的函数
check_error = \
  echo "Error: $1 failed"; \
  exit 1;

.PHONY: build-push-all

build-image:
	@echo "Building web Docker image: $(IMAGE_NAME):$(IMAGE_VERISON)..."
	docker buildx  build --platform $(PLATFORMS) -t $(IMAGE_NAME):$(IMAGE_VERISON) ./ || $(call check_error,"build-image")
	@echo "Web Docker image built successfully: $(IMAGE_NAME):$(IMAGE_VERISON)"

tag-image: build-image
	@echo "Taging web Docker image: $(IMAGE_NAME):$(IMAGE_VERISON)..."
	docker tag $(IMAGE_NAME):$(IMAGE_VERISON) $(REGISTRY_URL)/$(IMAGE_NAME):$(IMAGE_VERISON) || $(call check_error,"tag-image")
	@echo "Web Docker image tag successfully: $(IMAGE_NAME):$(IMAGE_VERISON)"

push-image: tag-image
	@echo "Pushing web Docker image: $(IMAGE_NAME):$(IMAGE_VERISON)..."
	docker push $(REGISTRY_URL)/$(IMAGE_NAME):$(IMAGE_VERISON) || $(call check_error,"push-image")
	@echo "Web Docker image push successfully: $(IMAGE_NAME):$(IMAGE_VERISON)"

run-image:
	docker stop $(CONTAINER_NAME) > /dev/null 2>&1 || true
	docker rm $(CONTAINER_NAME) > /dev/null 2>&1 || true
	@echo "Running web Docker container: $(CONTAINER_NAME)..."
	docker run -d --name $(CONTAINER_NAME) -p 8080:80 $(IMAGE_NAME):$(IMAGE_VERISON) || $(call check_error,"run-image")
	@echo "Web Docker container running successfully: $(CONTAINER_NAME)"

# 停止并清理Docker容器和镜像
clean-image:
	@echo "Cleaning web Docker container and image..."
	docker stop $(CONTAINER_NAME) > /dev/null 2>&1 || true
	docker rm $(CONTAINER_NAME) > /dev/null 2>&1 || true
	docker rmi -f $(IMAGE_NAME):$(IMAGE_VERISON) $(REGISTRY_URL)/$(IMAGE_NAME):$(IMAGE_VERISON) > /dev/null 2>&1 || true
	docker image prune -f -a > /dev/null 2>&1 || true
	@echo "Web Docker container and image cleaned successfully."

build-push-all: build-image tag-image push-image