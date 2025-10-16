"""Test Swedish climate region detection.

Verifies that the coordinator correctly detects Swedish climate regions
(Southern, Central, Northern, Lapland) based on GPS latitude coordinates.
"""

import pytest
from unittest.mock import Mock
from conftest import create_mock_hass, create_mock_entry
from custom_components.effektguard.coordinator import EffektGuardCoordinator
from custom_components.effektguard.const import (
    CLIMATE_SOUTHERN_SWEDEN,
    CLIMATE_CENTRAL_SWEDEN,
    CLIMATE_MID_NORTHERN_SWEDEN,
    CLIMATE_NORTHERN_SWEDEN,
    CLIMATE_NORTHERN_LAPLAND,
)


class TestSwedishClimateRegionDetection:
    """Test climate region detection for different Swedish latitudes."""

    @pytest.mark.asyncio
    async def test_detects_southern_sweden_malmo(self):
        """Test detection of southern Sweden region (Malmö, 55.6°N)."""
        mock_hass = create_mock_hass(latitude=55.6)  # Malmö
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_SOUTHERN_SWEDEN

    @pytest.mark.asyncio
    async def test_detects_southern_sweden_gothenburg(self):
        """Test detection of southern Sweden region (Gothenburg, 57.7°N)."""
        mock_hass = create_mock_hass(latitude=57.7)  # Gothenburg
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_SOUTHERN_SWEDEN

    @pytest.mark.asyncio
    async def test_detects_central_sweden_stockholm(self):
        """Test detection of central Sweden region (Stockholm, 59.3°N)."""
        mock_hass = create_mock_hass(latitude=59.3)  # Stockholm
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_CENTRAL_SWEDEN

    @pytest.mark.asyncio
    async def test_detects_central_sweden_uppsala(self):
        """Test detection of central Sweden region (Uppsala, 59.9°N)."""
        mock_hass = create_mock_hass(latitude=59.9)  # Uppsala
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_CENTRAL_SWEDEN

    @pytest.mark.asyncio
    async def test_detects_mid_northern_sweden_sundsvall(self):
        """Test detection of mid-northern Sweden region (Sundsvall, 62.4°N)."""
        mock_hass = create_mock_hass(latitude=62.4)  # Sundsvall
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_MID_NORTHERN_SWEDEN

    @pytest.mark.asyncio
    async def test_detects_northern_sweden_lulea(self):
        """Test detection of northern Sweden region (Luleå, 65.6°N)."""
        mock_hass = create_mock_hass(latitude=65.6)  # Luleå
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_NORTHERN_SWEDEN

    @pytest.mark.asyncio
    async def test_detects_lapland_kiruna(self):
        """Test detection of Lapland region (Kiruna, 67.9°N)."""
        mock_hass = create_mock_hass(latitude=67.9)  # Kiruna
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_NORTHERN_LAPLAND

    @pytest.mark.asyncio
    async def test_detects_lapland_abisko(self):
        """Test detection of Lapland region (Abisko, 68.4°N)."""
        mock_hass = create_mock_hass(latitude=68.4)  # Abisko
        mock_entry = create_mock_entry()

        coordinator = EffektGuardCoordinator(
            hass=mock_hass,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry,
        )

        assert coordinator.climate_region == CLIMATE_NORTHERN_LAPLAND


class TestClimateRegionBoundaries:
    """Test climate region detection at boundary latitudes."""

    @pytest.mark.asyncio
    async def test_boundary_southern_to_central(self):
        """Test boundary between Southern and Central Sweden (~58°N)."""
        # Just below boundary (Southern)
        mock_hass_south = create_mock_hass(latitude=57.9)
        mock_entry_south = create_mock_entry()
        coordinator_south = EffektGuardCoordinator(
            hass=mock_hass_south,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry_south,
        )
        assert coordinator_south.climate_region == CLIMATE_SOUTHERN_SWEDEN

        # Just above boundary (Central)
        mock_hass_central = create_mock_hass(latitude=58.1)
        mock_entry_central = create_mock_entry()
        coordinator_central = EffektGuardCoordinator(
            hass=mock_hass_central,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry_central,
        )
        assert coordinator_central.climate_region == CLIMATE_CENTRAL_SWEDEN

    @pytest.mark.asyncio
    async def test_boundary_central_to_mid_northern(self):  # renamed from central_to_northern
        """Test boundary between Central and Mid-Northern Sweden (~62°N)."""
        # Just below boundary (Central)
        mock_hass_central = create_mock_hass(latitude=60.9)
        mock_entry_central = create_mock_entry()
        coordinator_central = EffektGuardCoordinator(
            hass=mock_hass_central,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry_central,
        )
        assert coordinator_central.climate_region == CLIMATE_CENTRAL_SWEDEN

        # Just above boundary (still Central until 62.0)
        mock_hass_central2 = create_mock_hass(latitude=61.1)
        mock_entry_central2 = create_mock_entry()
        coordinator_central2 = EffektGuardCoordinator(
            hass=mock_hass_central2,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry_central2,
        )
        assert coordinator_central2.climate_region == CLIMATE_CENTRAL_SWEDEN

    @pytest.mark.asyncio
    async def test_boundary_northern_to_lapland(self):
        """Test boundary between Northern Sweden and Lapland (~67°N)."""
        # Just below boundary (Northern)
        mock_hass_northern = create_mock_hass(latitude=66.9)
        mock_entry_northern = create_mock_entry()
        coordinator_northern = EffektGuardCoordinator(
            hass=mock_hass_northern,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry_northern,
        )
        assert coordinator_northern.climate_region == CLIMATE_NORTHERN_SWEDEN

        # Just above boundary (Lapland)
        mock_hass_lapland = create_mock_hass(latitude=67.1)
        mock_entry_lapland = create_mock_entry()
        coordinator_lapland = EffektGuardCoordinator(
            hass=mock_hass_lapland,
            nibe_adapter=Mock(),
            gespot_adapter=Mock(),
            weather_adapter=Mock(),
            decision_engine=Mock(),
            effect_manager=Mock(),
            entry=mock_entry_lapland,
        )
        assert coordinator_lapland.climate_region == CLIMATE_NORTHERN_LAPLAND
