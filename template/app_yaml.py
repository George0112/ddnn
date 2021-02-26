app_yaml = """
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: %s
  namespace: default
  name: %s
spec:
  selector:
    matchLabels:
      app: %s
  replicas: 1
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: %s
    spec:
      containers:
      - image: chaowen/ddnn
        imagePullPolicy: Always
        name: %s
        resources:
          requests:
            cpu: "1000m"
          limits:
            cpu: "2000m"
        env:
          - name: "PYTHONUNBUFFERED"
            value: "1"
          - name: "PYTHONIOENCODING"
            value: "UTF-8"
        command: ["python3"]
        args: ["run.py", "%s", "%s", "%s", %s]
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      serviceAccount: ddnn-admin
"""

client_yaml = """apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: %s-client
  namespace: default
  name: %s-client
spec:
  progressDeadlineSeconds: 600
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: %s-client
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: %s-client
    spec:
      containers:
      - image: chaowen/vgg16:client
        imagePullPolicy: Always
        name: %s
        resources:
          requests:
            cpu: "500m"
          limits:
            cpu: "1000m"
        env:
          - name: "PYTHONUNBUFFERED"
            value: "1"
          - name: "PYTHONIOENCODING"
            value: "UTF-8"
        command: ["python3"]
        args: ["run.py", "%s", "%s"]
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
      terminationGracePeriodSeconds: 30
"""

svc_yaml = """apiVersion: v1
kind: Service
metadata:
  labels:
    app: %s
  name: %s
  namespace: default
spec:
  ports:
  - port: 5000
    protocol: TCP
    targetPort: 5000
  selector:
    app: %s
  sessionAffinity: None
  type: ClusterIP
"""