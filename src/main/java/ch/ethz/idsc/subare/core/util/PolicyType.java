// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.VsInterface;

public enum PolicyType {
  GREEDY() {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new EGreedyPolicy(discreteModel, qsa, sac).setExplorationRate(ConstantExplorationRate.of(0));
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new EGreedyPolicy(standardModel, vs, sac).setExplorationRate(ConstantExplorationRate.of(0));
    }
  }, //
  EGREEDY() {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new EGreedyPolicy(discreteModel, qsa, sac);
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new EGreedyPolicy(standardModel, vs, sac);
    }
  }, //
  UCB() {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new UcbPolicy(discreteModel, qsa, sac);
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new UcbPolicy(standardModel, vs, sac);
    }
  }, //
  ;
  abstract public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac);

  abstract public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac);
}
