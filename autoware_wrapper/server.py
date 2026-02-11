import os
from concurrent import futures
import grpc
from sbsvf_api import av_server_pb2_grpc
from sbsvf_api.pong_pb2 import Pong

import rclpy


class AVServer(av_server_pb2_grpc.AvServerServicer):
    def __init__(self):
        super().__init__()

    def Ping(self, request, context):
        print("Received Ping request")
        return Pong()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    av_server_pb2_grpc.add_AvServerServicer_to_server(AVServer(), server)

    PORT = os.environ.get("PORT", "50051")

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()

    print(f"gRPC server is running on port {PORT}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down gRPC server")
        server.stop(0)


if __name__ == "__main__":
    serve()
