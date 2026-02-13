#!/usr/bin/env python3
import argparse
import logging

from franka_sim import FrankaSimServer

# Configure logging to silence Numba debug output
logging.getLogger("numba").setLevel(logging.WARNING)


def main():
    """Run the Franka simulation server."""
    # get command line arguments visualization
    parser = argparse.ArgumentParser(description="Run a Franka simulation server")
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Enable visualization of the simulator",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    print(f"Starting Franka Simulation Server {'with' if args.vis else 'without'} visualization")
    print("Connect to the server using 'localhost' or '127.0.0.1' as the robot IP address")
    print("Press Ctrl+C to stop the server")

    server = FrankaSimServer(enable_vis=args.vis)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == "__main__":
    main()
