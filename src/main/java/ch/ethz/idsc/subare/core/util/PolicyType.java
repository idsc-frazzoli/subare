// code by jph
package ch.ethz.idsc.subare.core.util;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.VsInterface;

public enum PolicyType {
  GREEDY {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      EGreedyPolicy eGreedyPolicy = new EGreedyPolicy(discreteModel, qsa, sac);
      eGreedyPolicy.setExplorationRate(ConstantExplorationRate.of(0));
      return eGreedyPolicy;
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      EGreedyPolicy eGreedyPolicy = new EGreedyPolicy(standardModel, vs, sac);
      eGreedyPolicy.setExplorationRate(ConstantExplorationRate.of(0));
      return eGreedyPolicy;
    }
  },
  EGREEDY {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new EGreedyPolicy(discreteModel, qsa, sac);
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new EGreedyPolicy(standardModel, vs, sac);
    }
  },
  UCB {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new UcbPolicy(discreteModel, qsa, sac);
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new UcbPolicy(standardModel, vs, sac);
    }
  };

  public abstract PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac);

  public abstract PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac);
}
