from rest_framework import serializers

from recommender_profile.models import Address, UserProfile, TaskProfile


class AddressSerializer(serializers.ModelSerializer):
    class Meta:
        model = Address
        fields = ["country", "state", "city", "zip"]


class BaseProfileSerializer(serializers.ModelSerializer):
    address = AddressSerializer()  # Nested serializer for Address

    def create(self, validated_data):
        # Extract nested address data
        address_data = validated_data.pop("address")
        # Create the Address object
        address, _ = Address.objects.get_or_create(**address_data)
        validated_data.update(address=address)
        # Create the TaskProfile object with the associated Address
        instance = self.create_object(validated_data)
        return instance

    def create_object(self, validated_data):
        pass

    def update(self, instance, validated_data):
        # Handle nested address data
        address_data = validated_data.pop("address", None)
        if address_data:
            # Do not update the Address object, because Address may be related to another object
            address, _ = Address.objects.get_or_create(**address_data)
            instance.address = address

        # Update the Profile instance
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

class UserProfileSerializer(BaseProfileSerializer):
    class Meta:
        model = UserProfile
        fields = ["external_id", "skills", "is_available", "address"]

    def create_object(self, validated_data):
        return UserProfile.objects.create(**validated_data)


class TaskProfileSerializer(BaseProfileSerializer):
    class Meta:
        model = TaskProfile
        fields = ["external_id", "description", "is_available", "title", "skills", "address"]

    def create_object(self, validated_data):
        return TaskProfile.objects.create(**validated_data)
