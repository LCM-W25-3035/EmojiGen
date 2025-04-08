#!/bin/bash
BASE_URL="http://emojigen.ddns.net:8000"
ENV="front_end/.env"
sed -i "s|^REACT_APP_API_BASE_URL=.*|REACT_APP_API_BASE_URL=$BASE_URL|" "$ENV"