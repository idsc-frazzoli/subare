// code by jph
package ch.ethz.idsc.subare.ch04.rental;

import ch.ethz.idsc.subare.core.alg.ValueIteration;
import ch.ethz.idsc.subare.core.util.GreedyPolicy;
import ch.ethz.idsc.subare.util.UserHome;
import ch.ethz.idsc.tensor.io.GifSequenceWriter;
import ch.ethz.idsc.tensor.io.ImageFormat;

class CarRental_Ex4_02 {
  public static void main(String[] args) throws Exception {
    CarRental carRental = new CarRental();
    ValueIteration vi = new ValueIteration(carRental);
    GifSequenceWriter gsw = GifSequenceWriter.of(UserHome.file("Pictures/carrental_vi.gif"), 250);
    for (int count = 0; count < 10; ++count) {
      System.out.println(count);
      gsw.append(ImageFormat.of(CarRentalHelper.joinAll(carRental, vi.vs())));
      vi.step();
    }
    gsw.close();
    GreedyPolicy greedyPolicy = GreedyPolicy.bestEquiprobableGreedy(carRental, vi.vs());
    // Tensor values = vi.vs().values();
    // System.out.println(values);
    // // Put.of(UserHome.file("ex403_values"), values);
    // greedyPolicy.print(carRental.states());
    // // System.out.println(greedyPolicy.policy(RealScalar.of(49), RealScalar.of(1)));
    // Tensor greedy = greedyPolicy.flatten(carRental.states());
    // Put.of(UserHome.file("ex403_greedy"), greedy);
  }
}
