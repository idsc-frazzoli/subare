// code by fluric
package ch.ethz.idsc.subare.demo.airport;

import java.util.Arrays;

import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.alg.ActionValueIterations;
import ch.ethz.idsc.subare.core.mc.MonteCarloExploringStarts;
import ch.ethz.idsc.subare.core.td.OriginalSarsa;
import ch.ethz.idsc.subare.core.td.Sarsa;
import ch.ethz.idsc.subare.core.td.TrueOnlineSarsa;
import ch.ethz.idsc.subare.core.util.ConstantLearningRate;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.ExactFeatureMapper;
import ch.ethz.idsc.subare.core.util.ExploringStarts;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.core.util.StateActionCounter;
import ch.ethz.idsc.subare.util.AnalysisUtils;
import ch.ethz.idsc.subare.util.PlotUtils;
import ch.ethz.idsc.tensor.DecimalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** uses TrueOnlineSarsa */
enum Airport_Demo {
  ;
  public static void main(String[] args) throws Exception {
    Tensor XYmc = Tensors.empty();
    Tensor XYsarsa = Tensors.empty();
    Tensor XYtoSarsa = Tensors.empty();
    Airport airport = new Airport();
    DiscreteQsa optimalQsa = ActionValueIterations.solve(airport, DecimalScalar.of(.0001));
    // DiscreteUtils.print(optimalQsa);
    Policy policyQsa = GreedyPolicy.bestEquiprobable(airport, optimalQsa);
    // Policies.print(policyQsa, airport.states());
    int batches = 10000;
    MonteCarloExploringStarts mces = new MonteCarloExploringStarts(airport);
    for (int index = 0; index < batches; ++index) {
      Policy policyMC = EGreedyPolicy.bestEquiprobable(airport, mces.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(airport, policyMC, mces);
      XYmc.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(mces.qsa(), optimalQsa).number()));
    }
    // Policies.print(GreedyPolicy.bestEquiprobable(airport, mces.qsa()), airport.states());
    StateActionCounter sac = new StateActionCounter(airport);
    DiscreteQsa qsaSarsa = DiscreteQsa.build(airport); // q-function for training, initialized to 0
    final Sarsa sarsa = new OriginalSarsa(airport, qsaSarsa, ConstantLearningRate.of(RealScalar.of(0.05)));
    sarsa.setExplore(RealScalar.of(.1));
    for (int index = 0; index < batches; ++index) {
      Policy policy = EGreedyPolicy.bestEquiprobable(airport, sarsa.qsa(), RealScalar.of(.1));
      ExploringStarts.batch(airport, policy, 1, sarsa, sac);
      XYsarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number()));
    }
    // Policies.print(GreedyPolicy.bestEquiprobable(airport, sarsa.qsa()), airport.states());
    Tensor XY = Tensors.empty();
    FeatureMapper mapper = new ExactFeatureMapper(airport);
    TrueOnlineSarsa toSarsa = new TrueOnlineSarsa(airport, RealScalar.of(0.7), RealScalar.of(0.2), RealScalar.of(1), mapper);
    for (int index = 0; index < batches; ++index) {
      toSarsa.executeEpisode(RealScalar.of(0.1));
      DiscreteQsa toQsa = toSarsa.getQsa();
      XYtoSarsa.append(Tensors.vector(RealScalar.of(index).number(), AnalysisUtils.getLinearQsaError(toQsa, optimalQsa).number()));
    }
    DiscreteQsa toQsa = toSarsa.getQsa();
    // System.out.println(toSarsa.getW());
    // toSarsa.printValues();
    // toSarsa.printPolicy();
    System.out.println("Error of TrueOnlineSarsa: " + AnalysisUtils.getLinearQsaError(toQsa, optimalQsa).number().doubleValue() + " "
        + AnalysisUtils.getSquareQsaError(toQsa, optimalQsa).number().doubleValue());
    System.out.println("Error of Sarsa: " + AnalysisUtils.getLinearQsaError(sarsa.qsa(), optimalQsa).number().doubleValue() + " "
        + AnalysisUtils.getSquareQsaError(sarsa.qsa(), optimalQsa).number().doubleValue());
    System.out.println("Error of MonteCarlo: " + AnalysisUtils.getLinearQsaError(mces.qsa(), optimalQsa).number().doubleValue() + " "
        + AnalysisUtils.getSquareQsaError(mces.qsa(), optimalQsa).number().doubleValue());
    System.out.println(XY);
    PlotUtils.createPlot(Arrays.asList(XYmc, XYsarsa, XYtoSarsa), Arrays.asList("MonteCarlo", "Sarsa", "TrueOnlineSarsa"));
  }
}
