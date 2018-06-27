// code by fluric
package ch.ethz.idsc.subare.demo.airport;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.Policies;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;

enum Airport_Demo {
  ;
  public static void main(String[] args) throws Exception {
    Airport airport = new Airport();
    DiscreteQsa qsa = ActionValueIterations.solve(airport, DecimalScalar.of(.0001));
    DiscreteUtils.print(qsa);
    Policy policyQsa = GreedyPolicy.bestEquiprobable(airport, qsa);
    Policies.print(policyQsa, airport.states());
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(airport);
    int batches = 1000;
    for (int index = 0; index < batches; ++index) {
      Policy policyMC = EGreedyPolicy.bestEquiprobable(airport, mces.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(airport, policyMC, mces);
    }
    Policies.print(GreedyPolicy.bestEquiprobable(airport, mces.qsa()), airport.states());
    StateActionCounter sac = new StateActionCounter(airport);
    DiscreteQsa qsaSarsa = DiscreteQsa.build(airport); // q-function for training, initialized to 0
    final Sarsa sarsa = new OriginalSarsa(airport, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
    sarsa.setExplore(RealScalar.of(.1));
    for (int index = 0; index < batches; ++index) {
      Policy policy = EGreedyPolicy.bestEquiprobable(airport, sarsa.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(airport, policy, 1, sarsa, sac);
    }
    Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
    TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(airport, RealScalar.of(0.5), RealScalar.of(0.05), RealScalar.of(1));
    for (int index = 0; index < batches; ++index) {
      toSarsa.executeEpisode(RealScalar.of(0.1));
    }
    System.out.println(toSarsa.getW());
    toSarsa.printValues();
    toSarsa.printPolicy();
  }
}
