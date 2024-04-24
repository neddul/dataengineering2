import pulsar

# Use the IP address of the VM where Pulsar is running
client = pulsar.Client('pulsar://192.168.2.36:6650') #First IP, not associated one

# Create a producer on the topic that consumer can subscribe to
producer = client.create_producer('DEtopic')

# Send a message to consumer
producer.send(('Welcome to Data Engineering Course! 2B').encode('utf-8'))

# Destroy pulsar client
client.close()
