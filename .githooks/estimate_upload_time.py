import argparse
import os
import time

import speedtest


def run_speedtest():
    st_instance = speedtest.Speedtest()
    st_instance.get_best_server()
    upload_speed = st_instance.upload()
    return upload_speed


def main(pid, file_size, threshold, sleep_time):
    threshold = float(threshold)
    total_remaining_bytes = file_size
    last_upload_speed = 0
    start_time = time.time()

    while True:
        # Check if the process is still running
        try:
            os.kill(pid, 0)  # This just checks if the process is still running
        except OSError:
            print("Process finished.")
            break

        # Calculate time spent sleeping and account for bytes uploaded during this period
        current_time = time.time()
        elapsed_time = current_time - start_time
        start_time = current_time
        if last_upload_speed > 0:
            bytes_uploaded_during_sleep = last_upload_speed * elapsed_time
            total_remaining_bytes -= bytes_uploaded_during_sleep

        # Run a speed test and get the upload speed
        start_speedtest = time.time()
        upload_speed = run_speedtest()
        end_speedtest = time.time()

        # Calculate the duration of the speed test
        speedtest_duration = end_speedtest - start_speedtest

        # Estimate bytes uploaded during the speed test duration
        bytes_uploaded = upload_speed * speedtest_duration

        # Update the total remaining bytes
        total_remaining_bytes -= bytes_uploaded

        # Save the last upload speed
        last_upload_speed = upload_speed

        # Calculate the percentage remaining
        percentage_remaining = (total_remaining_bytes / file_size) * 100

        print(
            f"Current upload speed: {upload_speed} Total remaining bytes:"
            f" {total_remaining_bytes}, Percentage remaining:"
            f" {percentage_remaining:.2f}%"
        )

        # Check if the remaining bytes are within the allowed threshold
        # if total_remaining_bytes < -(threshold * file_size):
        #     print("Process exceeded threshold. Exiting monitoring.")
        #     break

        # Wait for a short duration before the next check
        time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Monitor upload progress based on speedtest results.'
    )
    parser.add_argument('--pid', type=int, help='Process ID to monitor')
    parser.add_argument(
        '--file_size', type=int, help='Total file size in bytes'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        help='Threshold percentage for allowed wiggle room',
    )
    parser.add_argument(
        '--sleep_time', type=int, help='Time to sleep between checks in seconds'
    )

    args = parser.parse_args()
    args.threshold = -float('inf')
    main(args.pid, args.file_size, args.threshold, args.sleep_time)
