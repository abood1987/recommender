from rest_framework import serializers

from recommender_profile.models import Address, UserProfile, TaskProfile


class AddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Address
        fields = ["country", "state", "city", "zip"]

class UserProfileSerializer(serializers.ModelSerializer):
    address = AddressSerializer()  # Nested serializer for Address

    class Meta:
        model = UserProfile
        fields = ["external_id", "skills", "is_available", "address"]

    def create(self, validated_data):
        # Extract nested address data
        address_data = validated_data.pop("address")
        # Create the Address object
        address = Address.objects.create(**address_data)
        # Create the UserProfile object with the associated Address
        user_profile = UserProfile.objects.create(address=address, **validated_data)
        return user_profile

    def update(self, instance, validated_data):
        # Handle nested address data
        address_data = validated_data.pop('address', None)
        if address_data:
            # Create allows a new Address object, because Address may be related to another object
            address = Address.objects.create(**address_data)
            instance.address = address

        # Update the TaskProfile instance
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        return instance


class TaskProfileSerializer(serializers.ModelSerializer):
    address = AddressSerializer()  # Nested serializer for Address

    class Meta:
        model = TaskProfile
        fields = ["external_id", "description", "is_available", "title", "skills", "address"]

    def create(self, validated_data):
        # Extract nested address data
        address_data = validated_data.pop("address")
        # Create the Address object
        address = Address.objects.create(**address_data)
        # Create the TaskProfile object with the associated Address
        task_profile = TaskProfile.objects.create(address=address, **validated_data)
        return task_profile

    def update(self, instance, validated_data):
        # Handle nested address data
        address_data = validated_data.pop('address', None)
        if address_data:
            # Create allows a new Address object, because Address may be related to another object
            address = Address.objects.create(**address_data)
            instance.address = address

        # Update the TaskProfile instance
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()

        return instance