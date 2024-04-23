import pulsar

client = pulsar.Client('pulsar://localhost:6650')


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
