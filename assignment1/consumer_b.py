import pulsar

# Use the IP address of the VM where Pulsar is running
client = pulsar.Client('pulsar://192.168.2.36:6650')

consumer = client.subscribe('DEtopic', subscription_name='DE-sub')

# Display message received from producer
msg = consumer.receive()

try:
    print("Received message : '%s'" % msg.data())

    # Acknowledge for receiving the message
    consumer.acknowledge(msg)
except:
    consumer.negative_acknowledge(msg)

# Destroy pulsar client
client.close()
