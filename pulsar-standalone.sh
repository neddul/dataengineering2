#!/bin/bash

# Run Apache Pulsar in standalone mode using Docker
docker run -it -p 6650:6650 -p 8080:8080 \
--mount source=pulsardata,target=/pulsar/data \
--mount source=pulsarconf,target=/pulsar/conf \
apachepulsar/pulsar:2.7.0 bin/pulsar standalone

