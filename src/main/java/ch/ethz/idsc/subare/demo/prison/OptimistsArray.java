// code by jph
package ch.ethz.idsc.subare.demo.prison;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import ch.ethz.idsc.subare.ch02.Agent;
import ch.ethz.idsc.subare.ch02.OptimistAgent;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.alg.ConstantArray;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.alg.Subdivide;
import ch.ethz.idsc.tensor.ext.HomeDirectory;
import ch.ethz.idsc.tensor.img.ColorDataGradients;
import ch.ethz.idsc.tensor.io.Export;
import ch.ethz.idsc.tensor.sca.N;

/* package */ class OptimistsArray {
  private final List<List<AbstractExact>> array = new ArrayList<>();

  public OptimistsArray(Tensor init, Scalar alpha) {
    for (Tensor q0 : init) {
      List<AbstractExact> list = new ArrayList<>();
      for (Tensor q1 : init) {
        Supplier<Agent> sup1 = //
            () -> new OptimistAgent(2, (Scalar) q0, alpha);
        Supplier<Agent> sup2 = //
            () -> new OptimistAgent(2, (Scalar) q1, alpha);
        list.add(new OptimistsExact(sup1, sup2));
      }
      array.add(list);
    }
  }

  private void play(int epochs) {
    array.forEach(list -> list.forEach(abstractExact -> abstractExact.play(epochs)));
  }

  public Tensor expectedRewards() {
    return Tensor.of(array.stream() //
        .map(list -> Tensor.of(list.stream() //
            .map(AbstractExact::getExpectedRewards))));
  }

  public Tensor actionReminder() {
    return Tensor.of(array.stream() //
        .map(list -> Tensor.of(list.stream() //
            .map(AbstractExact::getActionReminder))));
  }

  public static void main(String[] args) throws IOException {
    Tensor init = Subdivide.of(RationalScalar.of(+21, 10), RationalScalar.of(-11, 10), 280 - 1); //
    Tensor separator = ConstantArray.of(RealScalar.ZERO, init.length(), 5);
    Scalar alpha = RealScalar.of(0.22);
    OptimistsArray optimistsArray = new OptimistsArray(N.DOUBLE.of(init), alpha);
    File folder = HomeDirectory.Pictures(optimistsArray.getClass().getSimpleName() + "_" + alpha);
    folder.mkdir();
    for (int frame = 0; frame < 300; ++frame) {
      System.out.println("frame=" + frame);
      optimistsArray.play(1);
      Tensor tensor = optimistsArray.expectedRewards().get(Tensor.ALL, Tensor.ALL, 0);
      Tensor action = optimistsArray.actionReminder();
      // ScalarSummaryStatistics scalarSummaryStatistics = tensor.flatten(-1) //
      // .map(Scalar.class::cast) //
      // .collect(ScalarSummaryStatistics.collector());
      // System.out.println(scalarSummaryStatistics.toString());
      Tensor imageL = tensor.map(RealScalar.ONE::add).multiply(RationalScalar.HALF); //
      Tensor image = Join.of(1, imageL, separator, action).map(ColorDataGradients.CLASSIC);
      File file = new File(folder, String.format("%04d.png", frame));
      Export.of(file, image);
    }
  }
}
