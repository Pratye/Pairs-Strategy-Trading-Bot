#!/bin/bash

# Set the timezone to US Eastern
export TZ="America/New_York"

# Infinite loop to keep checking the time
while true; do
  # Get the current hour and minute in 24-hour format
  current_time=$(date +"%H:%M")

  # Define the start and end time in 24-hour format
  start_time="09:25"
  end_time="16:00"

  # Compare the current time with the working hours
  if [[ "$current_time" > "$start_time" && "$current_time" < "$end_time" ]]; then
    echo "Within business hours. Starting the application..."
    python main.py
  else
    echo "Outside business hours. Sleeping for 10 minutes..."
    sleep 600  # Sleep for 10 minutes before checking the time again
  fi
done
