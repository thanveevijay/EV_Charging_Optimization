# Timezone Handling

Store event timestamps with timezone information or in a documented UTC representation. Convert to local time only for display and calendar-based features.

Test daylight-saving transitions where relevant and avoid comparing naive timestamps from different locations.