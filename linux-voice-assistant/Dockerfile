ARG BUILD_FROM
FROM ${BUILD_FROM}

# Install Python and audio tools
RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    avahi-utils alsa-utils libportaudio2 portaudio19-dev \
    build-essential libmpv-dev python3 python3-venv python3-pip python3-dev \
    pulseaudio-utils libasound2-plugins && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . ./

RUN ./script/setup
RUN chmod +x ./script/run
CMD ["./script/run"]
