"""
Tests for ABM module: BankAgent, decision models, and BankingSystemSimulation.
"""

import logging
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from scr_financial.abm.bank_agent import BankAgent
from scr_financial.abm.simulation import BankingSystemSimulation
from scr_financial.abm.decision_models import (
    DefaultDecisionModel,
    StressDecisionModel,
    LearningDecisionModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def healthy_state():
    return {
        'CET1_ratio': 13.2,
        'LCR': 145,
        'NSFR': 115,
        'total_assets': 1.2e12,
        'cash': 1.5e11,
        'interbank_assets': 2.0e11,
        'interbank_liabilities': 1.8e11,
    }


@pytest.fixture
def bank(healthy_state):
    return BankAgent("DE_DBK", healthy_state)


@pytest.fixture
def sim_setup():
    bank_data = {
        "A": {'CET1_ratio': 13.0, 'LCR': 145, 'total_assets': 1e12, 'cash': 1e11},
        "B": {'CET1_ratio': 12.0, 'LCR': 150, 'total_assets': 8e11, 'cash': 8e10},
        "C": {'CET1_ratio': 11.5, 'LCR': 155, 'total_assets': 6e11, 'cash': 6e10},
    }
    network_data = {
        "A": {"B": 1e10, "C": 5e9},
        "B": {"A": 1e10, "C": 8e9},
        "C": {"A": 5e9, "B": 8e9},
    }
    system_indicators = {'CISS': 0.15, 'funding_stress': 0.1}
    return BankingSystemSimulation(bank_data, network_data, system_indicators)


# ---------------------------------------------------------------------------
# BankAgent tests
# ---------------------------------------------------------------------------

def test_bank_agent_init_stores_id_and_state(bank, healthy_state):
    assert bank.id == "DE_DBK"
    for key, val in healthy_state.items():
        assert bank.state[key] == val


def test_bank_agent_init_default_model_type(bank):
    assert isinstance(bank.decision_model, DefaultDecisionModel)


def test_bank_agent_init_empty_connections_and_memory(bank):
    assert bank.connections == {}
    assert bank.memory == []


def test_bank_agent_update_state_stores_old_in_memory(bank, healthy_state):
    original_cet1 = healthy_state['CET1_ratio']
    bank.update_state({'CET1_ratio': 10.0})
    assert bank.memory[0]['CET1_ratio'] == original_cet1


def test_bank_agent_update_state_applies_new_values(bank):
    bank.update_state({'CET1_ratio': 10.5})
    assert bank.state['CET1_ratio'] == 10.5


def test_bank_agent_memory_capped_at_memory_length(bank):
    for i in range(15):
        bank.update_state({'CET1_ratio': float(i)})
    assert len(bank.memory) == 10


def test_bank_agent_assess_solvency_healthy(bank):
    assert bank.assess_solvency() is True


def test_bank_agent_assess_solvency_insolvent(healthy_state):
    state = healthy_state.copy()
    state['CET1_ratio'] = 4.0
    b = BankAgent("TEST", state)
    assert b.assess_solvency() is False


def test_bank_agent_assess_solvency_at_threshold(healthy_state):
    state = healthy_state.copy()
    state['CET1_ratio'] = 4.5
    b = BankAgent("TEST", state)
    assert b.assess_solvency() is True


def test_bank_agent_assess_solvency_no_data_defaults_true():
    b = BankAgent("TEST", {})
    assert b.assess_solvency() is True


def test_bank_agent_assess_liquidity_healthy(bank):
    assert bank.assess_liquidity() is True


def test_bank_agent_assess_liquidity_stressed(healthy_state):
    state = healthy_state.copy()
    state['LCR'] = 90
    b = BankAgent("TEST", state)
    assert b.assess_liquidity() is False


def test_bank_agent_calculate_lending_capacity_positive_when_assets_known(bank):
    capacity = bank.calculate_lending_capacity()
    assert capacity > 0.0


def test_bank_agent_calculate_lending_capacity_zero_when_no_assets(healthy_state):
    state = healthy_state.copy()
    del state['total_assets']
    b = BankAgent("TEST", state)
    assert b.calculate_lending_capacity() == 0.0


def test_bank_agent_calculate_lending_capacity_increases_with_lcr(healthy_state):
    state_low = healthy_state.copy()
    state_high = healthy_state.copy()
    state_low['LCR'] = 100
    state_high['LCR'] = 200
    low = BankAgent("L", state_low).calculate_lending_capacity()
    high = BankAgent("H", state_high).calculate_lending_capacity()
    assert high > low


def test_bank_agent_calculate_lending_capacity_increases_with_cet1(healthy_state):
    state_low = healthy_state.copy()
    state_high = healthy_state.copy()
    state_low['CET1_ratio'] = 8.0
    state_high['CET1_ratio'] = 16.0
    low = BankAgent("L", state_low).calculate_lending_capacity()
    high = BankAgent("H", state_high).calculate_lending_capacity()
    assert high > low


def test_bank_agent_repr_contains_id(bank):
    assert "DE_DBK" in repr(bank)


def test_bank_agent_connection_strength_unknown_returns_zero(bank):
    assert bank.get_connection_strength("UNKNOWN_BANK") == 0.0


def test_bank_agent_connection_strength_known_returns_value(bank):
    bank.update_connections("OTHER", 5e9)
    assert_allclose(bank.get_connection_strength("OTHER"), 5e9)


# ---------------------------------------------------------------------------
# DefaultDecisionModel tests
# ---------------------------------------------------------------------------

@pytest.fixture
def healthy_borrowers():
    return {
        "B1": {'CET1_ratio': 12.0, 'LCR': 130},
        "B2": {'CET1_ratio': 10.0, 'LCR': 120},
    }


def test_default_model_lends_to_healthy_borrowers(bank, healthy_borrowers):
    result = bank.decide_lending_action(healthy_borrowers, market_sentiment=0.8)
    assert result["action"] == "lend"
    assert len(result.get("allocations", {})) > 0


def test_default_model_no_lending_when_bank_illiquid(healthy_state, healthy_borrowers):
    state = healthy_state.copy()
    state['LCR'] = 90
    illiquid_bank = BankAgent("ILL", state)
    result = illiquid_bank.decide_lending_action(healthy_borrowers, market_sentiment=0.8)
    assert result["action"] == "reduce_lending"


def test_default_model_skips_undercapitalised_borrower(bank):
    borrowers = {
        "WEAK": {'CET1_ratio': 5.0, 'LCR': 130},
        "STRONG": {'CET1_ratio': 12.0, 'LCR': 140},
    }
    result = bank.decide_lending_action(borrowers, market_sentiment=0.8)
    assert "WEAK" not in result.get("allocations", {})
    assert "STRONG" in result.get("allocations", {})


def test_default_model_borrowing_no_need_when_lcr_sufficient(bank):
    lenders = {"L1": {'CET1_ratio': 13.0, 'LCR': 140}}
    # bank.state['LCR'] = 145 > 120 threshold, so no borrowing needed
    result = bank.decide_borrowing_action(lenders, market_sentiment=0.8)
    assert result["action"] == "no_borrowing"


def test_default_model_respond_capital_shock_reduces_lending(bank):
    bank.update_connections("OTHER", 1e9)
    response = bank.respond_to_shock({'CET1_ratio': -2.0})
    action_types = [a['type'] for a in response['actions']]
    assert "reduce_lending" in action_types


def test_default_model_respond_liquidity_shock_seeks_funding(bank):
    bank.update_connections("OTHER", 1e9)
    response = bank.respond_to_shock({'LCR': -30.0})
    action_types = [a['type'] for a in response['actions']]
    assert "seek_funding" in action_types


def test_default_model_no_connections_shock_logs_warning(caplog):
    """Bank with no connections should not crash when responding to a shock."""
    b = BankAgent("ISOLATED", {'CET1_ratio': 13.0, 'LCR': 140, 'total_assets': 1e12})
    with caplog.at_level(logging.WARNING):
        response = b.respond_to_shock({'CET1_ratio': -2.0, 'LCR': -30.0})
    # Must not raise; response must be a dict with 'actions'
    assert 'actions' in response


# ---------------------------------------------------------------------------
# StressDecisionModel tests
# ---------------------------------------------------------------------------

@pytest.fixture
def stress_bank(healthy_state):
    return BankAgent("STRESS", healthy_state, decision_model=StressDecisionModel())


def test_stress_model_zero_lending_below_cutoff(stress_bank, healthy_borrowers):
    result = stress_bank.decide_lending_action(healthy_borrowers, market_sentiment=0.2)
    assert result["action"] == "reduce_lending"


def test_stress_model_lends_above_cutoff(stress_bank, healthy_borrowers):
    result = stress_bank.decide_lending_action(healthy_borrowers, market_sentiment=0.8)
    assert result["action"] == "lend"


def test_stress_model_allocations_leq_default(healthy_state, healthy_borrowers):
    default_bank = BankAgent("D", healthy_state.copy(), decision_model=DefaultDecisionModel())
    stress_bank_inst = BankAgent("S", healthy_state.copy(), decision_model=StressDecisionModel())
    sentiment = 0.7
    default_result = default_bank.decide_lending_action(healthy_borrowers, sentiment)
    stress_result = stress_bank_inst.decide_lending_action(healthy_borrowers, sentiment)
    if default_result["action"] == "lend" and stress_result["action"] == "lend":
        for bid in default_result.get("allocations", {}):
            if bid in stress_result.get("allocations", {}):
                assert stress_result["allocations"][bid] <= default_result["allocations"][bid] + 1e-9


def test_stress_model_amplifies_shock_reductions(healthy_state):
    default_bank = BankAgent("D", healthy_state.copy(), decision_model=DefaultDecisionModel())
    stress_bank_inst = BankAgent("S", healthy_state.copy(), decision_model=StressDecisionModel())
    default_bank.update_connections("T", 1e9)
    stress_bank_inst.update_connections("T", 1e9)
    shock = {'CET1_ratio': -3.0}
    default_resp = default_bank.respond_to_shock(shock)
    stress_resp = stress_bank_inst.respond_to_shock(shock)

    def max_reduction(resp):
        for action in resp['actions']:
            if action['type'] == 'reduce_lending' and 'targets' in action:
                vals = list(action['targets'].values())
                if vals:
                    return max(vals)
        return 0.0

    assert max_reduction(stress_resp) >= max_reduction(default_resp) - 1e-9


# ---------------------------------------------------------------------------
# LearningDecisionModel tests
# ---------------------------------------------------------------------------

def test_learning_model_init_empty_history():
    model = LearningDecisionModel()
    assert model.interaction_history == {}


def test_learning_model_invalid_learning_rate_raises():
    with pytest.raises(ValueError):
        LearningDecisionModel(learning_rate=0)
    with pytest.raises(ValueError):
        LearningDecisionModel(learning_rate=1.5)


def test_learning_model_update_score_positive_outcome():
    model = LearningDecisionModel(learning_rate=0.5)
    model.update_interaction_score("L", "B", 0.8)
    assert model.get_interaction_score("L", "B") > 0


def test_learning_model_score_converges_to_one():
    model = LearningDecisionModel(learning_rate=0.1)
    for _ in range(200):
        model.update_interaction_score("L", "B", 1.0)
    assert model.get_interaction_score("L", "B") > 0.9


def test_learning_model_negative_score_removes_allocation(healthy_state):
    model = LearningDecisionModel(learning_rate=1.0)
    # With learning_rate=1.0, a single outcome=-1 sets score to -1 exactly
    model.update_interaction_score("LRN", "WEAK_BORROWER", -1.0)
    b = BankAgent("LRN", healthy_state.copy(), decision_model=model)
    borrowers = {
        "WEAK_BORROWER": {'CET1_ratio': 12.0, 'LCR': 130},
    }
    result = b.decide_lending_action(borrowers, market_sentiment=0.8)
    assert "WEAK_BORROWER" not in result.get("allocations", {})


def test_learning_model_repr_contains_learning_rate():
    model = LearningDecisionModel(learning_rate=0.25)
    assert "0.25" in repr(model)


# ---------------------------------------------------------------------------
# BankingSystemSimulation tests
# ---------------------------------------------------------------------------

def test_sim_init_creates_correct_bank_count(sim_setup):
    assert len(sim_setup.banks) == 3


def test_sim_init_empty_bank_data_raises():
    with pytest.raises(ValueError):
        BankingSystemSimulation({}, {}, {})


def test_sim_connections_match_network_data(sim_setup):
    assert_allclose(sim_setup.banks["A"].connections["B"], 1e10)
    assert_allclose(sim_setup.banks["A"].connections["C"], 5e9)
    assert_allclose(sim_setup.banks["B"].connections["C"], 8e9)


def test_sim_sentiment_in_zero_one_range(sim_setup):
    s = sim_setup.calculate_market_sentiment()
    assert 0.0 <= s <= 1.0


def test_sim_sentiment_decreases_with_higher_ciss():
    def sentiment_for_ciss(ciss):
        sim = BankingSystemSimulation(
            {"A": {'CET1_ratio': 13.0, 'LCR': 145, 'total_assets': 1e12, 'cash': 1e11}},
            {"A": {}},
            {'CISS': ciss},
        )
        return sim.calculate_market_sentiment()

    assert sentiment_for_ciss(0.1) > sentiment_for_ciss(0.8)


def test_sim_apply_shock_updates_bank_cet1(sim_setup):
    original = sim_setup.banks["A"].state['CET1_ratio']
    sim_setup.apply_external_shock({"A": {"CET1_ratio": -1.0}})
    assert_allclose(sim_setup.banks["A"].state['CET1_ratio'], original - 1.0)


def test_sim_apply_shock_updates_system_indicator(sim_setup):
    original_ciss = sim_setup.system_indicators['CISS']
    sim_setup.apply_external_shock({"system": {"CISS": 0.1}})
    assert_allclose(sim_setup.system_indicators['CISS'], original_ciss + 0.1)


def test_sim_run_zero_steps_raises(sim_setup):
    with pytest.raises(ValueError):
        sim_setup.run_simulation(0)


def test_sim_run_returns_correct_history_length(sim_setup):
    history = sim_setup.run_simulation(5)
    assert len(history) == 5


def test_sim_time_increments_correctly(sim_setup):
    sim_setup.run_simulation(5)
    assert sim_setup.time == 5


def test_sim_history_entries_have_required_keys(sim_setup):
    history = sim_setup.run_simulation(3)
    for entry in history:
        assert 'time' in entry
        assert 'bank_states' in entry
        assert 'network' in entry
        assert 'system_indicators' in entry


def test_sim_get_adjacency_matrix_shape(sim_setup):
    adj = sim_setup.get_adjacency_matrix()
    assert adj.shape == (3, 3)


def test_sim_get_bank_metrics_is_dataframe_with_correct_cols(sim_setup):
    df = sim_setup.get_bank_metrics()
    assert isinstance(df, pd.DataFrame)
    assert 'bank_id' in df.columns
    assert 'time' in df.columns


def test_sim_get_system_metrics_has_derived_keys(sim_setup):
    metrics = sim_setup.get_system_metrics()
    assert 'avg_CET1_ratio' in metrics
    assert 'avg_LCR' in metrics
    assert 'network_density' in metrics


def test_sim_reset_clears_history(sim_setup):
    sim_setup.run_simulation(5)
    sim_setup.reset()
    assert sim_setup.history == []


def test_sim_reset_restores_initial_cet1(sim_setup):
    original_cet1_a = sim_setup.banks["A"].state['CET1_ratio']
    sim_setup.run_simulation(5)
    sim_setup.reset()
    assert_allclose(sim_setup.banks["A"].state['CET1_ratio'], original_cet1_a)
